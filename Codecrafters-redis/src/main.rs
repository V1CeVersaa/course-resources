mod commands;
mod parser;
mod replication;
mod storage;
mod utils;

use clap::Parser;
use replication::{generate_replication_id, parse_replicaof, ReplicationClient, ServerRole};
use std::sync::{Arc, Mutex};
use storage::MemoryStorage;
use tokio::net::TcpListener;
use utils::{handle_connection, ReplicaSenders, SharedStorage};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 6379)]
    port: u16,
    #[arg(long)]
    replicaof: Option<String>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    println!("Logs from your program will appear here!");

    let server_role = match args.replicaof {
        Some(replicaof_str) => match parse_replicaof(replicaof_str.as_str()) {
            Ok((master_host, master_port)) => {
                println!("Starting as replica of {}:{}", master_host, master_port);

                ServerRole::Slave {
                    master_host,
                    master_port,
                }
            }
            Err(e) => {
                eprintln!("Failed to parse replicaof: {}", e);
                let replication_id = generate_replication_id();
                println!("Starting as master with replication ID {}", replication_id);
                ServerRole::Master {
                    replication_id,
                    replication_offset: 0,
                }
            }
        },
        None => {
            let replication_id = generate_replication_id();
            println!("Starting as master with replication ID {}", replication_id);
            ServerRole::Master {
                replication_id,
                replication_offset: 0,
            }
        }
    };

    let storage: SharedStorage = Arc::new(Mutex::new(MemoryStorage::new()));
    // If started as replica, kick off the replication read loop now that storage exists.
    if let ServerRole::Slave {
        master_host,
        master_port,
    } = server_role.clone()
    {
        let storage_clone = storage.clone();
        let replica_port = args.port;
        let master_host_cl = master_host.clone();
        tokio::spawn(async move {
            match ReplicationClient::connect(&master_host_cl, master_port, replica_port).await {
                Ok(mut client) => {
                    if let Err(e) = client.perform_handshake().await {
                        eprintln!("Handshake failed: {}", e);
                        return;
                    }
                    if let Err(e) = client.receive_rdb_and_run(storage_clone).await {
                        eprintln!("Replication loop error: {}", e);
                    }
                }
                Err(e) => eprintln!("Failed to connect to master: {}", e),
            }
        });
    }

    let listener = TcpListener::bind(format!("127.0.0.1:{}", args.port)).await.unwrap();
    println!("Redis Server listening on port {}", args.port);

    let replica_senders: ReplicaSenders = Arc::new(utils::ReplicationState {
        senders: Mutex::new(Vec::new()),
        ack_counter: std::sync::atomic::AtomicU64::new(0),
        connected: std::sync::atomic::AtomicU64::new(0),
        writes_sent: std::sync::atomic::AtomicU64::new(0),
        master_offset: std::sync::atomic::AtomicU64::new(0),
    });

    loop {
        let (socket, _) = listener.accept().await.unwrap();
        let storage_clone = storage.clone();
        let server_role_clone = server_role.clone();
        let replica_senders_clone = replica_senders.clone();

        tokio::spawn(async move {
            handle_connection(socket, storage_clone, server_role_clone, replica_senders_clone).await;
        });
    }
}
