use crate::parser::{parse_command_with_consumed, RespValue};
use crate::storage::Storage;
use rand::Rng;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[derive(Debug, Clone)]
pub enum ServerRole {
    Master {
        replication_id: String,
        replication_offset: u64,
    },
    Slave {
        master_host: String,
        master_port: u16,
    },
}

pub struct ReplicationClient {
    stream: TcpStream,
    replica_port: u16,
}

impl ReplicationClient {
    pub async fn connect(master_host: &str, master_port: u16, replica_port: u16) -> Result<Self, String> {
        let stream = TcpStream::connect(format!("{}:{}", master_host, master_port))
            .await
            .map_err(|e| format!("Failed to connect to master: {}", e))?;
        println!("Connected to master at {}:{}", master_host, master_port);

        Ok(Self { stream, replica_port })
    }

    pub async fn perform_handshake(&mut self) -> Result<(), String> {
        self.send_ping().await?;
        self.read_response().await?; // +PONG

        self.send_replconf_listening_port().await?;
        self.read_response().await?; // +OK

        self.send_replconf_capabilities().await?;
        self.read_response().await?; // +OK

        self.send_psync().await?;
        println!("Handshake completed successfully (waiting for RDB)");
        Ok(())
    }

    async fn send_ping(&mut self) -> Result<(), String> {
        let command = encode_resp_array(&["PING".to_string()]);
        self.stream
            .write_all(command.as_bytes())
            .await
            .map_err(|e| format!("Failed to send PING: {}", e))?;
        println!("Sent PING to master");
        Ok(())
    }

    async fn send_replconf_listening_port(&mut self) -> Result<(), String> {
        let command = encode_resp_array(&[
            "REPLCONF".to_string(),
            "listening-port".to_string(),
            self.replica_port.to_string(),
        ]);
        self.stream
            .write_all(command.as_bytes())
            .await
            .map_err(|e| format!("Failed to send REPLCONF listening-port: {}", e))?;
        println!("Sent REPLCONF listening-port {} to master", self.replica_port);
        Ok(())
    }

    async fn send_replconf_capabilities(&mut self) -> Result<(), String> {
        let command = encode_resp_array(&["REPLCONF".to_string(), "capa".to_string(), "psync2".to_string()]);
        self.stream
            .write_all(command.as_bytes())
            .await
            .map_err(|e| format!("Failed to send REPLCONF capa: {}", e))?;
        println!("Sent REPLCONF capa psync2");
        Ok(())
    }

    async fn send_psync(&mut self) -> Result<(), String> {
        let command = encode_resp_array(&["PSYNC".to_string(), "?".to_string(), "-1".to_string()]);
        self.stream
            .write_all(command.as_bytes())
            .await
            .map_err(|e| format!("Failed to send PSYNC: {}", e))?;
        println!("Sent PSYNC ? -1");
        Ok(())
    }

    async fn read_response(&mut self) -> Result<String, String> {
        let mut buffer = [0; 1024];
        let bytes_read = self
            .stream
            .read(&mut buffer)
            .await
            .map_err(|e| format!("Failed to read response: {}", e))?;
        let response = String::from_utf8_lossy(&buffer[..bytes_read]).to_string();
        println!("Received from master: {:?}", response.trim());
        Ok(response)
    }

    pub async fn receive_rdb_and_run(mut self, storage: Arc<Mutex<dyn Storage>>) -> Result<(), String> {
        let mut pending: Vec<u8> = Vec::new();

        // Read until we get the RDB bulk length and body
        // Expect: +FULLRESYNC ...\r\n$<len>\r\n<data>
        loop {
            let mut buf = [0u8; 4096];
            let n = self
                .stream
                .read(&mut buf)
                .await
                .map_err(|e| format!("Failed to read: {}", e))?;
            if n == 0 {
                return Err("Master closed while sending RDB".to_string());
            }
            pending.extend_from_slice(&buf[..n]);

            // Find "$<len>\r\n"
            if let Some(dollar_pos) = pending.iter().position(|&b| b == b'$') {
                if let Some(crlf_pos) = pending[dollar_pos..].windows(2).position(|w| w == b"\r\n") {
                    let start = dollar_pos + 1;
                    let end = dollar_pos + crlf_pos;
                    if let Ok(len_str) = std::str::from_utf8(&pending[start..end]) {
                        if let Ok(len) = len_str.parse::<usize>() {
                            let body_start = end + 2; // skip CRLF
                            if pending.len() >= body_start + len {
                                // Consume up to end of body
                                let remaining = pending.split_off(body_start + len);
                                pending = remaining; // keep only remaining after RDB
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Now process propagated commands in pending + future bytes
        self.run_replica_apply_loop(storage, pending).await
    }

    async fn run_replica_apply_loop(
        mut self,
        storage: Arc<Mutex<dyn Storage>>,
        mut pending: Vec<u8>,
    ) -> Result<(), String> {
        let mut buf = [0u8; 4096];
        let mut processed_bytes: usize = 0;
        loop {
            // Try parse as many commands as possible
            while let Ok((cmd, consumed)) = parse_command_with_consumed(&pending) {
                // REPLCONF GETACK * -> reply with ACK <offset>
                if cmd.len() >= 3
                    && cmd[0].eq_ignore_ascii_case("REPLCONF")
                    && cmd[1].eq_ignore_ascii_case("GETACK")
                {
                    let offset = processed_bytes;
                    let resp =
                        encode_resp_array(&["REPLCONF".to_string(), "ACK".to_string(), offset.to_string()]);
                    let _ = self.stream.write_all(resp.as_bytes()).await;
                } else {
                    apply_set_only(&cmd, storage.clone());
                }
                pending.drain(0..consumed);
                processed_bytes += consumed;
            }

            let n = self
                .stream
                .read(&mut buf)
                .await
                .map_err(|e| format!("Failed to read: {}", e))?;
            if n == 0 {
                return Ok(());
            }
            pending.extend_from_slice(&buf[..n]);
        }
    }
}

fn encode_resp_array(args: &[String]) -> String {
    let mut array = Vec::new();
    for arg in args {
        array.push(RespValue::BulkString(Some(arg.clone())));
    }
    RespValue::Array(Some(array)).to_string_response()
}

fn apply_set_only(command: &[String], storage: Arc<Mutex<dyn Storage>>) {
    if command.is_empty() {
        return;
    }
    if !command[0].eq_ignore_ascii_case("SET") {
        return;
    }
    if command.len() < 3 {
        return;
    }
    let _ = storage.lock().unwrap().set(command[1].clone(), command[2].clone());
    if command.len() == 5 && command[3].eq_ignore_ascii_case("PX") {
        if let Ok(ms) = command[4].parse::<u64>() {
            let _ = storage.lock().unwrap().expire(&command[1], ms);
        }
    }
}

pub fn generate_replication_id() -> String {
    const CHARSET: &[u8] = b"0123456789abcdef";
    const ID_LEN: usize = 40;

    let mut rng = rand::rng();
    (0..ID_LEN)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

pub fn parse_replicaof(replicaof_str: &str) -> Result<(String, u16), String> {
    let parts: Vec<&str> = replicaof_str.split_whitespace().collect();
    if parts.len() != 2 {
        return Err("Invalid replicaof format, expected '<host> <port>'".to_string());
    }

    let host = parts[0].to_string();
    let port = parts[1]
        .parse::<u16>()
        .map_err(|_| "Invalid port number in replicaof".to_string())?;

    Ok((host, port))
}
