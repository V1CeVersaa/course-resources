use crate::parser::RespValue;
use crate::replication::ServerRole;
use crate::storage::{BlockedClient, BlockedStreamClient, Storage};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

// Type alias for complex xread response type
type XReadResponse = Vec<(String, Vec<(String, Vec<String>)>)>;

const EMPTY_RDB_HEX: &str = "524544495330303131fa0972656469732d76657205372e322e30fa0a72656469732d62697473c040fa056374696d65c26d08bc65fa08757365642d6d656dc2b0c41000fa08616f662d62617365c000fff06e3bfec0ff5aa2";

pub struct CommandHandler {
    storage: Arc<Mutex<dyn Storage>>,
    in_transaction: bool,
    queued_commands: Vec<Vec<String>>,
    server_role: ServerRole,
}

#[derive(Debug)]
pub enum CommandResponse {
    Resp(RespValue),
    RdbFile(Vec<u8>),
}

impl CommandResponse {
    pub fn to_resp_value(&self) -> Option<RespValue> {
        match self {
            CommandResponse::Resp(resp) => Some(resp.clone()),
            CommandResponse::RdbFile(_) => None,
        }
    }
}

pub mod constants {
    pub const PING: &str = "PING";
    pub const ECHO: &str = "ECHO";
    pub const SET: &str = "SET";
    pub const GET: &str = "GET";
    pub const OK: &str = "OK";
    pub const PONG: &str = "PONG";
    pub const RPUSH: &str = "RPUSH";
    pub const LPUSH: &str = "LPUSH";
    pub const LPOP: &str = "LPOP";
    pub const BLPOP: &str = "BLPOP";
    pub const LRANGE: &str = "LRANGE";
    pub const LLEN: &str = "LLEN";
    pub const TYPE: &str = "TYPE";
    pub const XADD: &str = "XADD";
    pub const XRANGE: &str = "XRANGE";
    pub const XREAD: &str = "XREAD";
    pub const INCR: &str = "INCR";
    pub const MULTI: &str = "MULTI";
    pub const EXEC: &str = "EXEC";
    pub const DISCARD: &str = "DISCARD";
    pub const INFO: &str = "INFO";
    pub const REPLCONF: &str = "REPLCONF";
    pub const PSYNC: &str = "PSYNC";
}

impl CommandHandler {
    pub fn new(storage: Arc<Mutex<dyn Storage>>, server_role: ServerRole) -> Self {
        Self {
            storage,
            in_transaction: false,
            queued_commands: Vec::new(),
            server_role,
        }
    }

    pub async fn handle(&mut self, command: Vec<String>) -> CommandResponse {
        if command.is_empty() {
            return CommandResponse::Resp(RespValue::Error("ERR empty command".to_string()));
        }

        match command[0].as_str() {
            constants::MULTI => return self.handle_multi(&command),
            constants::EXEC => return self.handle_exec().await,
            constants::DISCARD => return self.handle_discard(&command),
            constants::PSYNC => return self.handle_psync(&command),
            _ => {}
        }

        if self.in_transaction {
            self.queued_commands.push(command);
            return CommandResponse::Resp(RespValue::SimpleString("QUEUED".to_string()));
        }

        match command[0].as_str() {
            constants::PING => self.handle_ping(&command),
            constants::ECHO => self.handle_echo(&command),
            constants::SET => self.handle_set(&command),
            constants::GET => self.handle_get(&command),
            constants::RPUSH => self.handle_rpush(&command),
            constants::LPUSH => self.handle_lpush(&command),
            constants::LRANGE => self.handle_lrange(&command),
            constants::LLEN => self.handle_llen(&command),
            constants::LPOP => self.handle_lpop(&command),
            constants::BLPOP => self.handle_blpop(&command).await,
            constants::TYPE => self.handle_type(&command),
            constants::XADD => self.handle_xadd(&command),
            constants::XRANGE => self.handle_xrange(&command),
            constants::XREAD => self.handle_xread(&command).await,
            constants::INCR => self.handle_incr(&command),
            constants::INFO => self.handle_info(&command),
            constants::REPLCONF => self.handle_replconf(&command),
            constants::PSYNC => self.handle_psync(&command),
            _ => CommandResponse::Resp(RespValue::Error(format!("ERR unknown command '{}'", command[0]))),
        }
    }

    /// address Ping command
    fn handle_ping(&self, _command: &[String]) -> CommandResponse {
        let ret = if _command.len() == 1 {
            RespValue::SimpleString(constants::PONG.to_string())
        } else if _command.len() == 2 {
            RespValue::BulkString(Some(_command[1].clone()))
        } else {
            RespValue::Error("ERR wrong number of arguments for 'ping' command".to_string())
        };
        CommandResponse::Resp(ret)
    }

    /// address Echo command
    fn handle_echo(&self, command: &[String]) -> CommandResponse {
        let ret = if command.len() == 2 {
            RespValue::BulkString(Some(command[1].clone()))
        } else {
            RespValue::Error("ERR wrong number of arguments for 'echo' command".to_string())
        };
        CommandResponse::Resp(ret)
    }

    /// address Set command
    fn handle_set(&mut self, command: &[String]) -> CommandResponse {
        let ret = if command.len() == 3 {
            self.storage
                .lock()
                .unwrap()
                .set(command[1].clone(), command[2].clone())
                .unwrap();
            RespValue::SimpleString(constants::OK.to_string())
        } else if command.len() == 5 {
            self.storage
                .lock()
                .unwrap()
                .set(command[1].clone(), command[2].clone())
                .unwrap();
            if command[3].to_uppercase() == "PX" {
                match command[4].parse::<u64>() {
                    Ok(ttl) => {
                        self.storage.lock().unwrap().expire(&command[1], ttl).unwrap();
                        RespValue::SimpleString(constants::OK.to_string())
                    }
                    Err(_) => RespValue::Error("ERR invalid TTL value".to_string()),
                }
            } else {
                RespValue::Error("ERR invalid expiration time".to_string())
            }
        } else {
            RespValue::Error("ERR wrong number of arguments for 'set' command".to_string())
        };
        CommandResponse::Resp(ret)
    }

    /// address Get command
    fn handle_get(&mut self, command: &[String]) -> CommandResponse {
        let ret = if command.len() == 2 {
            match self.storage.lock().unwrap().get(&command[1]) {
                Some(value) => RespValue::BulkString(Some(value)),
                None => RespValue::BulkString(None),
            }
        } else {
            RespValue::Error("ERR wrong number of arguments for 'get' command".to_string())
        };
        CommandResponse::Resp(ret)
    }

    fn handle_rpush(&mut self, command: &[String]) -> CommandResponse {
        if command.len() < 3 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'rpush' command".to_string(),
            ));
        }

        let key = command[1].clone();
        let mut list_length = 0;
        for value in &command[2..] {
            match self.storage.lock().unwrap().rpush(key.clone(), value.clone()) {
                Ok(len) => list_length = len,
                Err(e) => return CommandResponse::Resp(RespValue::Error(e)),
            }
        }
        CommandResponse::Resp(RespValue::Integer(list_length))
    }

    fn handle_lpush(&mut self, command: &[String]) -> CommandResponse {
        if command.len() < 3 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'lpush' command".to_string(),
            ));
        }

        let key = command[1].clone();
        let mut list_length = 0;
        for value in &command[2..] {
            match self.storage.lock().unwrap().lpush(key.clone(), value.clone()) {
                Ok(len) => list_length = len,
                Err(e) => return CommandResponse::Resp(RespValue::Error(e)),
            }
        }
        CommandResponse::Resp(RespValue::Integer(list_length))
    }

    fn handle_lrange(&self, command: &[String]) -> CommandResponse {
        if command.len() == 4 {
            let key = &command[1];

            let start = match command[2].parse::<i64>() {
                Ok(val) => val,
                Err(_) => return CommandResponse::Resp(RespValue::Error("ERR invalid start index".to_string())),
            };

            let stop = match command[3].parse::<i64>() {
                Ok(val) => val,
                Err(_) => return CommandResponse::Resp(RespValue::Error("ERR invalid stop index".to_string())),
            };
            let elements = self
                .storage
                .lock()
                .unwrap()
                .lrange(key, start, stop)
                .into_iter()
                .map(|s| RespValue::BulkString(Some(s)))
                .collect();

            CommandResponse::Resp(RespValue::Array(Some(elements)))
        } else {
            CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'lrange' command".to_string(),
            ))
        }
    }

    fn handle_llen(&self, command: &[String]) -> CommandResponse {
        if command.len() == 2 {
            let key = &command[1];
            let length = self.storage.lock().unwrap().llen(key);
            CommandResponse::Resp(RespValue::Integer(length))
        } else {
            CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'llen' command".to_string(),
            ))
        }
    }

    fn handle_lpop(&mut self, command: &[String]) -> CommandResponse {
        if command.len() == 2 {
            let key = &command[1];
            let elements = self.storage.lock().unwrap().lpop(key, None);
            match elements.len() {
                0 => CommandResponse::Resp(RespValue::BulkString(None)),
                1 => CommandResponse::Resp(RespValue::BulkString(Some(elements[0].clone()))),
                _ => CommandResponse::Resp(RespValue::Error(
                    "ERR zero parameter lpop returned multiple elements, expected one".to_string(),
                )),
            }
        } else if command.len() == 3 {
            let key = &command[1];
            let count = match command[2].parse::<usize>() {
                Ok(0) => {
                    return CommandResponse::Resp(RespValue::Error("ERR value is not a positive integer".to_string()))
                }
                Ok(val) => Some(val),
                Err(_) => return CommandResponse::Resp(RespValue::Error("ERR value is not an integer".to_string())),
            };
            let elements = self.storage.lock().unwrap().lpop(key, count);
            if elements.len() != count.unwrap() {
                return CommandResponse::Resp(RespValue::Error(
                    "ERR count is greater than the list length".to_string(),
                ));
            }
            let resp_elements: Vec<RespValue> = elements.into_iter().map(|s| RespValue::BulkString(Some(s))).collect();
            CommandResponse::Resp(RespValue::Array(Some(resp_elements)))
        } else {
            CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'lpop' command".to_string(),
            ))
        }
    }

    async fn handle_blpop(&mut self, command: &[String]) -> CommandResponse {
        if command.len() < 2 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'blpop' command".to_string(),
            ));
        }

        let timeout_str = &command[command.len() - 1];
        let keys: Vec<String> = command[1..command.len() - 1].to_vec();

        if keys.is_empty() {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'blpop' command".to_string(),
            ));
        }

        let timeout_secs = match timeout_str.parse::<f64>() {
            Ok(t) if t < 0.0 => {
                return CommandResponse::Resp(RespValue::Error("ERR timeout is not a positive number".to_string()))
            }
            Ok(t) => t,
            Err(_) => return CommandResponse::Resp(RespValue::Error("ERR timeout is not a valid float".to_string())),
        };

        if let Some((key, value)) = self.storage.lock().unwrap().try_blpop(&keys) {
            return CommandResponse::Resp(RespValue::Array(Some(vec![
                RespValue::BulkString(Some(key)),
                RespValue::BulkString(Some(value)),
            ])));
        }

        let (tx, mut rx) = mpsc::unbounded_channel();
        let client_id = self.storage.lock().unwrap().generate_client_id();

        let timeout = if timeout_secs == 0.0 {
            None
        } else {
            Some(Duration::from_secs_f64(timeout_secs))
        };

        let blocked_client = BlockedClient {
            client_id,
            list_keys: keys.clone(),
            timeout,
            start_time: Instant::now(),
            response_sender: tx,
        };

        self.storage.lock().unwrap().add_blocked_client(blocked_client);

        let timeout_duration = timeout.unwrap_or(Duration::from_secs(u64::MAX));

        match tokio::time::timeout(timeout_duration, rx.recv()).await {
            Ok(Some(Some((key, value)))) => CommandResponse::Resp(RespValue::Array(Some(vec![
                RespValue::BulkString(Some(key)),
                RespValue::BulkString(Some(value)),
            ]))),
            _ => {
                self.cleanup_blocked_client(client_id, &keys);
                CommandResponse::Resp(RespValue::Array(None))
            }
        }
    }

    fn cleanup_blocked_client(&mut self, client_id: u64, keys: &[String]) {
        self.storage.lock().unwrap().cleanup_blocked_client(client_id, keys);
    }

    fn handle_type(&mut self, command: &[String]) -> CommandResponse {
        if command.len() != 2 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'type' command".to_string(),
            ));
        }

        let key = &command[1];
        let type_name = self.storage.lock().unwrap().get_type(key);
        CommandResponse::Resp(RespValue::SimpleString(type_name))
    }

    fn handle_xadd(&mut self, command: &[String]) -> CommandResponse {
        if command.len() < 4 || (command.len() - 3) % 2 != 0 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'xadd' command".to_string(),
            ));
        }

        let key = &command[1];
        let id = &command[2];

        let mut fields = Vec::new();
        for i in (3..command.len()).step_by(2) {
            let field = command[i].clone();
            let value = command[i + 1].clone();
            fields.push((field, value));
        }

        match self.storage.lock().unwrap().xadd(key.clone(), id.to_string(), fields) {
            Ok(resp) => CommandResponse::Resp(RespValue::BulkString(Some(resp))),
            Err(err) => CommandResponse::Resp(RespValue::Error(err)),
        }
    }

    fn handle_xrange(&mut self, command: &[String]) -> CommandResponse {
        if command.len() != 4 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'xrange' command".to_string(),
            ));
        }

        let key = &command[1];
        let start = &command[2];
        let end = &command[3];

        match self.storage.lock().unwrap().xrange(key, start, end) {
            Ok(entries) => {
                let mut result = Vec::new();
                for (id, fields) in entries {
                    let mut entry_array = Vec::new();
                    let fields_array: Vec<RespValue> =
                        fields.into_iter().map(|f| RespValue::BulkString(Some(f))).collect();

                    entry_array.push(RespValue::BulkString(Some(id)));
                    entry_array.push(RespValue::Array(Some(fields_array)));
                    result.push(RespValue::Array(Some(entry_array)));
                }
                CommandResponse::Resp(RespValue::Array(Some(result)))
            }
            Err(err) => CommandResponse::Resp(RespValue::Error(err)),
        }
    }

    async fn handle_xread(&mut self, command: &[String]) -> CommandResponse {
        if command.len() < 4 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'xread' command".to_string(),
            ));
        }

        let mut block_timeout: Option<Duration> = None;
        let mut args_start = 1;

        if command.len() > 2 && command[1].to_uppercase() == "BLOCK" {
            if command.len() < 5 {
                return CommandResponse::Resp(RespValue::Error(
                    "ERR wrong number of arguments for 'xread' command".to_string(),
                ));
            }

            let timeout_ms: u64 = match command[2].parse() {
                Ok(ms) => ms,
                Err(_) => return CommandResponse::Resp(RespValue::Error("ERR invalid timeout".to_string())),
            };

            block_timeout = if timeout_ms == 0 {
                None
            } else {
                Some(Duration::from_millis(timeout_ms))
            };

            args_start = 3;
        }

        let streams_pos = command[args_start..].iter().position(|x| x.to_uppercase() == "STREAMS");
        if streams_pos.is_none() {
            return CommandResponse::Resp(RespValue::Error("ERR syntax error".to_string()));
        }

        let streams_index = args_start + streams_pos.unwrap();
        let remaining_args = &command[streams_index + 1..];
        if remaining_args.len() < 2 || remaining_args.len() % 2 != 0 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'xread' command".to_string(),
            ));
        }

        let mut streams = Vec::new();
        let stream_count = remaining_args.len() / 2;
        let keys = &remaining_args[0..stream_count];
        let ids = &remaining_args[stream_count..];

        for i in 0..stream_count {
            let key = keys[i].clone();
            let id = if ids[i] == "$" {
                match self.storage.lock().unwrap().get_max_stream_id(&key) {
                    Some(max_id) => max_id,
                    None => "0-0".to_string(),
                }
            } else {
                ids[i].clone()
            };
            streams.push((key, id));
        }

        match self.storage.lock().unwrap().xread(streams.clone()) {
            Ok(results) if !results.is_empty() => {
                return self.format_xread_response(results);
            }
            Ok(_) => {
                if block_timeout.is_none() && args_start == 1 {
                    return CommandResponse::Resp(RespValue::Array(Some(Vec::new())));
                } // not a blocking call, return empty array
            }
            Err(err) => return CommandResponse::Resp(RespValue::Error(err)),
        }

        // If blocking is requested, set up blocking client
        if block_timeout.is_some() || args_start > 1 {
            let (tx, mut rx) = mpsc::unbounded_channel();

            let client_id = {
                let mut storage = self.storage.lock().unwrap();
                storage.generate_stream_client_id()
            };

            let blocked_client = BlockedStreamClient {
                client_id,
                streams: streams.clone(),
                timeout: block_timeout,
                start_time: Instant::now(),
                response_sender: tx,
            };

            {
                let mut storage = self.storage.lock().unwrap();
                storage.add_blocked_stream_client(blocked_client);
            }

            // Wait for response or timeout
            let result = if let Some(timeout) = block_timeout {
                tokio::time::timeout(timeout, rx.recv()).await
            } else {
                Ok(rx.recv().await)
            };

            {
                let mut storage = self.storage.lock().unwrap();
                storage.cleanup_blocked_stream_client(client_id, &streams);
            }

            match result {
                Ok(Some(Some(data))) => self.format_xread_response(data),
                _ => CommandResponse::Resp(RespValue::Array(None)),
            }
        } else {
            CommandResponse::Resp(RespValue::Array(Some(Vec::new())))
        }
    }

    fn format_xread_response(&self, results: XReadResponse) -> CommandResponse {
        let mut resp = Vec::new();
        for (key, entries) in results {
            let mut stream_array = Vec::new();
            let mut entries_array = Vec::new();
            for (id, fields) in entries {
                let mut entry_array = Vec::new();
                let fields_array: Vec<RespValue> = fields.into_iter().map(|f| RespValue::BulkString(Some(f))).collect();

                entry_array.push(RespValue::BulkString(Some(id)));
                entry_array.push(RespValue::Array(Some(fields_array)));
                entries_array.push(RespValue::Array(Some(entry_array)));
            }
            stream_array.push(RespValue::BulkString(Some(key)));
            stream_array.push(RespValue::Array(Some(entries_array)));
            resp.push(RespValue::Array(Some(stream_array)));
        }
        CommandResponse::Resp(RespValue::Array(Some(resp)))
    }

    fn handle_incr(&mut self, command: &[String]) -> CommandResponse {
        if command.len() != 2 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'incr' command".to_string(),
            ));
        }

        let key = &command[1];

        match self.storage.lock().unwrap().incr(key) {
            Ok(value) => CommandResponse::Resp(RespValue::Integer(value)),
            Err(err) => CommandResponse::Resp(RespValue::Error(err)),
        }
    }

    fn handle_multi(&mut self, command: &[String]) -> CommandResponse {
        if command.len() != 1 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'multi' command".to_string(),
            ));
        }

        if self.in_transaction {
            return CommandResponse::Resp(RespValue::Error("ERR MULTI calls can not be nested".to_string()));
        }

        self.in_transaction = true;
        self.queued_commands.clear();
        CommandResponse::Resp(RespValue::SimpleString("OK".to_string()))
    }

    async fn handle_exec(&mut self) -> CommandResponse {
        if !self.in_transaction {
            return CommandResponse::Resp(RespValue::Error("ERR EXEC without MULTI".to_string()));
        }

        let commands = std::mem::take(&mut self.queued_commands);
        self.in_transaction = false;

        let mut results = Vec::new();
        for cmd in commands {
            let result = self.execute_single_command(cmd).await;
            results.push(
                result
                    .to_resp_value()
                    .unwrap_or(RespValue::Error("ERR command failed".to_string())),
            );
        }

        CommandResponse::Resp(RespValue::Array(Some(results)))
    }

    async fn execute_single_command(&mut self, command: Vec<String>) -> CommandResponse {
        if command.is_empty() {
            return CommandResponse::Resp(RespValue::Error("ERR empty command".to_string()));
        }

        match command[0].as_str() {
            constants::PING => self.handle_ping(&command),
            constants::ECHO => self.handle_echo(&command),
            constants::SET => self.handle_set(&command),
            constants::GET => self.handle_get(&command),
            constants::RPUSH => self.handle_rpush(&command),
            constants::LPUSH => self.handle_lpush(&command),
            constants::LRANGE => self.handle_lrange(&command),
            constants::LLEN => self.handle_llen(&command),
            constants::LPOP => self.handle_lpop(&command),
            constants::BLPOP => self.handle_blpop(&command).await,
            constants::TYPE => self.handle_type(&command),
            constants::XADD => self.handle_xadd(&command),
            constants::XRANGE => self.handle_xrange(&command),
            constants::XREAD => self.handle_xread(&command).await,
            constants::INCR => self.handle_incr(&command),
            constants::INFO => self.handle_info(&command),
            constants::REPLCONF => self.handle_replconf(&command),
            constants::PSYNC => self.handle_psync(&command),
            _ => CommandResponse::Resp(RespValue::Error(format!("ERR unknown command '{}'", command[0]))),
        }
    }

    fn handle_discard(&mut self, command: &[String]) -> CommandResponse {
        if command.len() != 1 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'discard' command".to_string(),
            ));
        }

        if !self.in_transaction {
            return CommandResponse::Resp(RespValue::Error("ERR DISCARD without MULTI".to_string()));
        }

        self.in_transaction = false;
        self.queued_commands.clear();
        CommandResponse::Resp(RespValue::SimpleString("OK".to_string()))
    }

    fn handle_info(&self, command: &[String]) -> CommandResponse {
        if command.len() == 1 {
            self.get_replication_info()
        } else if command.len() == 2 {
            let section = command[1].to_uppercase();
            match section.to_uppercase().as_str() {
                "REPLICATION" => self.get_replication_info(),
                _ => CommandResponse::Resp(RespValue::Error(format!("ERR unknown INFO section '{}'", section))),
            }
        } else {
            CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'info' command".to_string(),
            ))
        }
    }

    fn handle_replconf(&self, command: &[String]) -> CommandResponse {
        if command.len() < 2 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'replconf' command".to_string(),
            ));
        }

        match command.len() {
            3 => println!("Received REPLCONF {} {}", command[1], command[2]),
            2 => println!("Received REPLCONF {}", command[1]),
            _ => println!("Received REPLCONF with {} arguments", command.len()),
        }

        CommandResponse::Resp(RespValue::SimpleString("OK".to_string()))
    }

    fn handle_psync(&self, command: &[String]) -> CommandResponse {
        if command.len() != 3 {
            return CommandResponse::Resp(RespValue::Error(
                "ERR wrong number of arguments for 'psync' command".to_string(),
            ));
        }

        let repl_id = &command[1];
        let offset = &command[2];

        println!("Received PSYNC {} {}", repl_id, offset);

        if repl_id == "?" && offset == "-1" {
            match &self.server_role {
                ServerRole::Master {
                    replication_id,
                    replication_offset,
                } => {
                    let mut response_data = Vec::new();

                    let fullresync_response = format!("FULLRESYNC {} {}", replication_id, replication_offset);
                    let fullresync_resp = RespValue::SimpleString(fullresync_response);
                    response_data.extend_from_slice(fullresync_resp.to_string_response().as_bytes());

                    let empty_rdb = hex_to_bytes(EMPTY_RDB_HEX);
                    let rdb_reader = format!("${}\r\n", empty_rdb.len());
                    response_data.extend_from_slice(rdb_reader.as_bytes());
                    response_data.extend_from_slice(&empty_rdb);

                    CommandResponse::RdbFile(response_data)
                }
                ServerRole::Slave { .. } => {
                    CommandResponse::Resp(RespValue::Error("ERR slave cannot handle PSYNC command".to_string()))
                }
            }
        } else {
            CommandResponse::Resp(RespValue::Error("ERR incremental sync not supported yet".to_string()))
        }
    }

    fn get_replication_info(&self) -> CommandResponse {
        let info = match &self.server_role {
            ServerRole::Master {
                replication_id,
                replication_offset,
            } => {
                format!(
                    "role:master\r\nmaster_replid:{}\r\nmaster_repl_offset:{}\r\n",
                    replication_id, replication_offset
                )
            }
            ServerRole::Slave {
                master_host,
                master_port,
            } => {
                format!(
                    "role:slave\r\nmaster_host:{}\r\nmaster_port:{}\r\n",
                    master_host, master_port
                )
            }
        };
        CommandResponse::Resp(RespValue::BulkString(Some(info)))
    }
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}
