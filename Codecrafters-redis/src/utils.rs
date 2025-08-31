use crate::commands::{CommandHandler, CommandResponse};
use crate::parser::{parse_command, parse_command_with_consumed, RespValue};
use crate::replication::ServerRole;
use crate::storage::Storage;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration, Instant};

// Type alias for complex sender type
type ReplicaSender = (mpsc::UnboundedSender<Vec<u8>>, Arc<AtomicU64>);

pub type SharedStorage = Arc<Mutex<dyn Storage>>;

pub struct ReplicationState {
    pub senders: Mutex<Vec<ReplicaSender>>, // (tx, last_acked)
    pub ack_counter: AtomicU64, // legacy counter, not required for offset-based WAIT
    pub connected: AtomicU64,
    pub writes_sent: AtomicU64,
    pub master_offset: AtomicU64,
}

pub type ReplicaSenders = Arc<ReplicationState>;

fn encode_resp_array(args: &[String]) -> Vec<u8> {
    let mut arr = Vec::new();
    for a in args {
        arr.push(RespValue::BulkString(Some(a.clone())));
    }
    RespValue::Array(Some(arr)).to_string_response().into_bytes()
}

pub async fn handle_connection(
    mut stream: TcpStream,
    storage: SharedStorage,
    server_role: ServerRole,
    replica_senders: ReplicaSenders,
) {
    let mut buffer = [0; 1024];
    let mut command_handler = CommandHandler::new(storage.clone(), server_role.clone());
    let mut replica_counted = false;
    let mut last_write_offset_for_this_client: u64 = 0;

    loop {
        match stream.read(&mut buffer).await {
            Ok(0) => {
                break;
            }
            Ok(bytes_read) => match parse_command(&buffer[..bytes_read]) {
                Ok(command) => {
                    let cmd_clone = command.clone();
                    // Handle WAIT here (master only)
                    if let ServerRole::Master { .. } = server_role {
                        // Count replica on REPLCONF listening-port (first handshake signal)
                        if !replica_counted
                            && cmd_clone.len() >= 3
                            && cmd_clone[0].eq_ignore_ascii_case("REPLCONF")
                            && cmd_clone[1].eq_ignore_ascii_case("listening-port")
                        {
                            replica_senders.connected.fetch_add(1, Ordering::SeqCst);
                            replica_counted = true;
                        }
                        if !cmd_clone.is_empty() && cmd_clone[0].eq_ignore_ascii_case("WAIT") {
                            let response =
                                handle_wait_command(&cmd_clone, &replica_senders, last_write_offset_for_this_client)
                                    .await;
                            if let Err(e) = stream.write_all(response.as_bytes()).await {
                                println!("Failed to write WAIT response: {}", e);
                                break;
                            }
                            continue;
                        }
                    }

                    let command_response: CommandResponse = command_handler.handle(command).await;

                    match command_response {
                        CommandResponse::Resp(resp_value) => {
                            let response_str = resp_value.to_string_response();
                            if let Err(e) = stream.write_all(response_str.as_bytes()).await {
                                println!("Failed to write response: {}", e);
                                break;
                            }

                            // After responding to client, broadcast SET if master
                            if let ServerRole::Master { .. } = server_role {
                                if !cmd_clone.is_empty() && cmd_clone[0].eq_ignore_ascii_case("SET") {
                                    let bytes = encode_resp_array(&cmd_clone);
                                    let len = bytes.len() as u64;
                                    let senders = replica_senders.senders.lock().unwrap().clone();
                                    for (tx, _) in &senders {
                                        let _ = tx.send(bytes.clone());
                                    }
                                    replica_senders.writes_sent.fetch_add(1, Ordering::SeqCst);
                                    let new_offset =
                                        replica_senders.master_offset.fetch_add(len, Ordering::SeqCst) + len;
                                    last_write_offset_for_this_client = new_offset;
                                }
                            }
                        }
                        CommandResponse::RdbFile(rdb_data) => {
                            if let Err(e) = stream.write_all(&rdb_data).await {
                                println!("Failed to write RDB file response: {}", e);
                                break;
                            }
                            println!("Sent FULLRESYNC and RDB file ({} bytes total)", rdb_data.len());

                            // Register this connection as a replica target: spawn writer task and return
                            let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
                            let rep_ack = Arc::new(AtomicU64::new(0));
                            {
                                let mut list = replica_senders.senders.lock().unwrap();
                                list.push((tx, rep_ack.clone()));
                            }
                            let (mut read_half, mut write_half) = stream.into_split();
                            let state_for_read = replica_senders.clone();
                            if !replica_counted {
                                state_for_read.connected.fetch_add(1, Ordering::SeqCst);
                            }
                            tokio::spawn(async move {
                                while let Some(bytes) = rx.recv().await {
                                    if let Err(e) = write_half.write_all(&bytes).await {
                                        eprintln!("Failed to write to replica: {}", e);
                                        break;
                                    }
                                }
                            });
                            tokio::spawn(async move {
                                let mut pending: Vec<u8> = Vec::new();
                                let mut buf = [0u8; 4096];
                                loop {
                                    let n = read_half.read(&mut buf).await.unwrap_or_default();
                                    if n == 0 {
                                        break;
                                    }
                                    pending.extend_from_slice(&buf[..n]);
                                    while let Ok((cmd, consumed)) = parse_command_with_consumed(&pending) {
                                        if cmd.len() >= 2
                                            && cmd[0].eq_ignore_ascii_case("REPLCONF")
                                            && cmd[1].eq_ignore_ascii_case("ACK")
                                        {
                                            if cmd.len() >= 3 {
                                                if let Ok(off) = cmd[2].parse::<u64>() {
                                                    rep_ack.store(off, Ordering::SeqCst);
                                                } else {
                                                    state_for_read.ack_counter.fetch_add(1, Ordering::SeqCst);
                                                }
                                            } else {
                                                state_for_read.ack_counter.fetch_add(1, Ordering::SeqCst);
                                            }
                                        }
                                        pending.drain(0..consumed);
                                    }
                                }
                            });
                            return;
                        }
                    }
                }
                Err(e) => {
                    println!("Failed to parse command: {}", e);
                    let error_response = RespValue::Error(format!("ERR {}", e));
                    if let Err(e) = stream.write_all(error_response.to_string_response().as_bytes()).await {
                        println!("Failed to write error response: {}", e);
                        break;
                    }
                }
            },

            Err(e) => {
                println!("error: {}", e);
                break;
            }
        }
    }
}

async fn handle_wait_command(args: &[String], state: &ReplicaSenders, client_write_offset: u64) -> String {
    if args.len() != 3 {
        return RespValue::Error("ERR wrong number of arguments for 'wait' command".to_string()).to_string_response();
    }
    let wanted: i64 = args[1].parse().unwrap_or(0);
    let timeout_ms: u64 = args[2].parse().unwrap_or(0);

    // Compute connected replicas
    let by_counter = state.connected.load(Ordering::SeqCst) as i64;
    let by_senders = state.senders.lock().unwrap().len() as i64;
    let replica_count = std::cmp::max(by_counter, by_senders);
    if replica_count == 0 {
        return RespValue::Integer(0).to_string_response();
    }

    // Ask all replicas for ACK and wait up to timeout
    let getack = encode_resp_array(&["REPLCONF".to_string(), "GETACK".to_string(), "*".to_string()]);
    {
        let list = state.senders.lock().unwrap().clone();
        for (tx, _) in list {
            let _ = tx.send(getack.clone());
        }
    }

    let target = std::cmp::min(wanted, replica_count);
    if target <= 0 {
        return RespValue::Integer(replica_count).to_string_response();
    }

    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        if Instant::now() >= deadline {
            break;
        }
        // Count replicas whose last_acked >= client_write_offset
        let ok = {
            let list = state.senders.lock().unwrap();
            let mut count = 0i64;
            for (_, last) in list.iter() {
                if last.load(Ordering::SeqCst) >= client_write_offset {
                    count += 1;
                }
            }
            count
        };
        if ok >= target {
            break;
        }
        sleep(Duration::from_millis(5)).await;
    }
    let ok = {
        let list = state.senders.lock().unwrap();
        let mut count = 0i64;
        for (_, last) in list.iter() {
            if last.load(Ordering::SeqCst) >= client_write_offset {
                count += 1;
            }
        }
        count
    };
    RespValue::Integer(ok).to_string_response()
}
