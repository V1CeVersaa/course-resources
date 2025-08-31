use super::{BasicStorage, BlockedClient, BlockedStreamClient, ListStorage, StreamEntry, StreamStorage};
use crate::storage::stream::StreamId;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
pub enum RedisValue {
    String(String),
    List(Vec<String>),
    Stream(Vec<StreamEntry>),
}

pub trait Storage: BasicStorage + ListStorage + StreamStorage {}

/// memory storage implementation, using HashMap to store data, support expiration time
pub struct MemoryStorage {
    data: HashMap<String, RedisValue>,
    expiry: HashMap<String, SystemTime>,
    blocked_clients: HashMap<String, Vec<BlockedClient>>,
    client_counter: u64,
    blocked_stream_clients: HashMap<String, Vec<BlockedStreamClient>>,
    next_stream_client_id: u64,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            expiry: HashMap::new(),
            blocked_clients: HashMap::new(),
            client_counter: 0,
            blocked_stream_clients: HashMap::new(),
            next_stream_client_id: 0,
        }
    }

    fn is_expired(&self, key: &str) -> bool {
        if let Some(expiry_time) = self.expiry.get(key) {
            SystemTime::now() > *expiry_time
        } else {
            false
        }
    }

    fn remove_if_expired(&mut self, key: &str) -> bool {
        if self.is_expired(key) {
            self.data.remove(key);
            self.expiry.remove(key);
            true
        } else {
            false
        }
    }
}

impl BasicStorage for MemoryStorage {
    fn set(&mut self, key: String, value: String) -> Result<(), String> {
        self.data.insert(key.clone(), RedisValue::String(value)); // insert key-value pair
        self.expiry.remove(&key); // remove key expiration time
        Ok(())
    }

    fn get(&self, key: &str) -> Option<String> {
        if self.is_expired(key) {
            return None;
        }
        match self.data.get(key) {
            Some(RedisValue::String(s)) => Some(s.clone()),
            Some(RedisValue::List(_)) => None,
            Some(RedisValue::Stream(_)) => None,
            None => None,
        }
    }

    fn expire(&mut self, key: &str, milli_seconds: u64) -> Result<bool, String> {
        if self.data.contains_key(key) {
            self.expiry.insert(
                key.to_string(),
                SystemTime::now() + Duration::from_millis(milli_seconds),
            );
            Ok(true)
        } else {
            Err("ERR key not found".to_string())
        }
    }

    fn get_type(&self, key: &str) -> String {
        if self.is_expired(key) {
            return "none".to_string();
        }

        match self.data.get(key) {
            Some(RedisValue::String(_)) => "string".to_string(),
            Some(RedisValue::List(_)) => "list".to_string(),
            Some(RedisValue::Stream(_)) => "stream".to_string(),
            None => "none".to_string(),
        }
    }

    fn incr(&mut self, key: &str) -> Result<i64, String> {
        self.remove_if_expired(key);

        match self.data.get(key) {
            Some(RedisValue::String(value)) => {
                // Try to parse existing value as integer
                match value.parse::<i64>() {
                    Ok(num) => {
                        let new_value = num + 1;
                        self.data
                            .insert(key.to_string(), RedisValue::String(new_value.to_string()));
                        self.expiry.remove(key); // Remove expiration when updating
                        Ok(new_value)
                    }
                    Err(_) => Err("ERR value is not an integer or out of range".to_string()),
                }
            }
            Some(_) => {
                // Key exists but is not a string (could be list, stream, etc.)
                Err("WRONGTYPE Operation against a key holding the wrong kind of value".to_string())
            }
            None => {
                // Key doesn't exist, set to 1
                self.data.insert(key.to_string(), RedisValue::String("1".to_string()));
                Ok(1)
            }
        }
    }
}

impl ListStorage for MemoryStorage {
    fn rpush(&mut self, key: String, value: String) -> Result<i64, String> {
        self.remove_if_expired(key.as_str());

        let result = match self.data.get_mut(key.as_str()) {
            None => {
                let new_list = vec![value];
                self.data.insert(key.clone(), RedisValue::List(new_list));
                Ok(1)
            }
            Some(RedisValue::List(list)) => {
                list.push(value);
                Ok(list.len() as i64)
            }
            Some(_) => Err("WRONGTYPE Operation against a key holding the wrong kind of value".to_string()),
        };

        if result.is_ok() {
            self.notify_blocked_clients(&key);
        }

        result
    }

    fn lpush(&mut self, key: String, value: String) -> Result<i64, String> {
        self.remove_if_expired(key.as_str());

        match self.data.get_mut(key.as_str()) {
            None => {
                let new_list = vec![value];
                self.data.insert(key, RedisValue::List(new_list));
                Ok(1)
            }
            Some(RedisValue::List(list)) => {
                list.insert(0, value);
                Ok(list.len() as i64)
            }
            Some(_) => Err("WRONGTYPE Operation against a key holding the wrong kind of value".to_string()),
        }
    }

    fn llen(&self, key: &str) -> i64 {
        if self.is_expired(key) {
            return 0;
        }
        match self.data.get(key) {
            Some(RedisValue::List(list)) => list.len() as i64,
            Some(_) => 0,
            None => 0,
        }
    }

    fn lrange(&self, key: &str, start: i64, stop: i64) -> Vec<String> {
        if self.is_expired(key) {
            return Vec::<String>::new();
        }

        match self.data.get(key) {
            Some(RedisValue::List(list)) => {
                let len = list.len() as i64;
                if len == 0 {
                    return Vec::<String>::new();
                }

                let actual_start = if start < 0 { len + start } else { start };
                let actual_stop = if stop < 0 { len + stop } else { stop };

                if actual_start >= len || actual_stop < 0 || actual_start > actual_stop {
                    return Vec::<String>::new();
                }

                let start_index = if actual_start < 0 { 0 } else { actual_start } as usize;
                let stop_index = if actual_stop >= len { len } else { actual_stop + 1 } as usize;

                list[start_index..stop_index].to_vec()
            }
            Some(_) => Vec::<String>::new(),
            None => Vec::<String>::new(),
        }
    }

    fn lpop(&mut self, key: &str, count: Option<usize>) -> Vec<String> {
        if self.is_expired(key) {
            self.remove_if_expired(key);
            return Vec::<String>::new();
        }

        let count = count.unwrap_or(1);

        match self.data.get_mut(key) {
            Some(RedisValue::List(list)) => {
                let actual_count = std::cmp::min(count, list.len());
                let popped_elements: Vec<String> = list.drain(0..actual_count).collect();
                if list.is_empty() {
                    self.data.remove(key);
                    self.expiry.remove(key);
                }
                popped_elements
            }
            Some(_) => Vec::<String>::new(),
            None => Vec::<String>::new(),
        }
    }

    fn try_blpop(&mut self, keys: &[String]) -> Option<(String, String)> {
        for key in keys {
            self.remove_if_expired(key);
            if let Some(RedisValue::List(list)) = self.data.get_mut(key) {
                if !list.is_empty() {
                    let value = list.remove(0);
                    if list.is_empty() {
                        self.data.remove(key);
                        self.expiry.remove(key);
                    }
                    return Some((key.clone(), value));
                }
            }
        }
        None
    }

    fn add_blocked_client(&mut self, client: BlockedClient) {
        for key in &client.list_keys {
            self.blocked_clients
                .entry(key.clone())
                .or_default()
                .push(client.clone());
        }
    }

    fn notify_blocked_clients(&mut self, key: &str) {
        let (client_id, client_keys) = {
            let Some(clients) = self.blocked_clients.get_mut(key) else {
                return;
            };
            if clients.is_empty() {
                return;
            }

            clients.sort_unstable_by(|a, b| a.start_time.cmp(&b.start_time));

            let Some(RedisValue::List(list)) = self.data.get_mut(key) else {
                return;
            };
            if list.is_empty() {
                return;
            }

            let value = list.remove(0);
            if list.is_empty() {
                self.data.remove(key);
                self.expiry.remove(key);
            }

            let Some(client) = clients.first() else { return };
            let _ = client.response_sender.send(Some((key.to_string(), value)));
            (client.client_id, client.list_keys.clone())
        };

        for client_key in &client_keys {
            if let Some(key_clients) = self.blocked_clients.get_mut(client_key) {
                key_clients.retain(|c| c.client_id != client_id);
                if key_clients.is_empty() {
                    self.blocked_clients.remove(client_key);
                }
            }
        }
    }

    fn generate_client_id(&mut self) -> u64 {
        self.client_counter += 1;
        self.client_counter
    }

    fn cleanup_blocked_client(&mut self, client_id: u64, keys: &[String]) {
        for key in keys {
            if let Some(clients) = self.blocked_clients.get_mut(key) {
                clients.retain(|c| c.client_id != client_id);
                if clients.is_empty() {
                    self.blocked_clients.remove(key);
                }
            }
        }
    }
}

impl StreamStorage for MemoryStorage {
    fn xadd(&mut self, key: String, id: String, fields: Vec<(String, String)>) -> Result<String, String> {
        self.remove_if_expired(key.as_str());

        let last_id = match self.data.get(key.as_str()) {
            Some(RedisValue::Stream(stream)) => stream.last().map(|entry| &entry.id),
            _ => None,
        };

        let stream_id = StreamId::parse(&id, last_id)?;

        if !stream_id.is_greater_than_zero() {
            return Err("ERR The ID specified in XADD must be greater than 0-0".to_string());
        }

        let entry = StreamEntry {
            id: stream_id.clone(),
            fields,
        };

        match self.data.get_mut(key.as_str()) {
            None => {
                let stream = vec![entry];
                self.data.insert(key.clone(), RedisValue::Stream(stream));
                self.notify_blocked_stream_clients(&key);
                Ok(stream_id.to_string())
            }
            Some(RedisValue::Stream(stream)) => {
                if let Some(last_entry) = stream.last() {
                    if last_entry.id >= stream_id {
                        return Err(
                            "ERR The ID specified in XADD is equal or smaller than the target stream top item"
                                .to_string(),
                        );
                    }
                }
                stream.push(entry);
                self.notify_blocked_stream_clients(&key);
                Ok(stream_id.to_string())
            }
            Some(_) => Err("WRONGTYPE Operation against a key holding the wrong kind of value".to_string()),
        }
    }

    fn xrange(&self, key: &str, start: &str, end: &str) -> Result<Vec<(String, Vec<String>)>, String> {
        if self.is_expired(key) {
            return Ok(Vec::new());
        }

        match self.data.get(key) {
            Some(RedisValue::Stream(stream)) => {
                let start_id = StreamId::parse_range_id(start, 0)?;
                let end_id = StreamId::parse_range_id(end, u64::MAX)?;

                let mut result = Vec::new();
                for entry in stream {
                    if entry.id.is_in_range(&start_id, &end_id) {
                        let mut fields = Vec::new();
                        for (field, value) in &entry.fields {
                            fields.push(field.clone());
                            fields.push(value.clone());
                        }
                        result.push((entry.id.to_string(), fields));
                    }
                }
                Ok(result)
            }
            Some(_) => Err("WRONGTYPE Operation against a key holding the wrong kind of value".to_string()),
            None => Ok(Vec::new()),
        }
    }

    fn xread(&self, streams: Vec<(String, String)>) -> Result<Vec<(String, Vec<(String, Vec<String>)>)>, String> {
        let mut result = Vec::new();

        for (key, start_id_str) in streams {
            if self.is_expired(key.as_str()) {
                continue;
            }

            match self.data.get(key.as_str()) {
                Some(RedisValue::Stream(stream)) => {
                    let start_id = StreamId::parse(&start_id_str, None).map_err(|_| "Invalid stream ID".to_string())?;

                    let mut entries = Vec::new();
                    for entry in stream {
                        if entry.id > start_id {
                            let mut fields = Vec::new();
                            for (field_key, value) in &entry.fields {
                                fields.push(field_key.clone());
                                fields.push(value.clone());
                            }
                            entries.push((entry.id.to_string(), fields));
                        }
                    }

                    if !entries.is_empty() {
                        result.push((key, entries));
                    }
                }
                Some(_) => return Err("WRONGTYPE Operation against a key holding the wrong kind of value".to_string()),
                None => continue,
            }
        }
        Ok(result)
    }

    fn get_max_stream_id(&self, key: &str) -> Option<String> {
        if self.is_expired(key) {
            return None;
        }

        match self.data.get(key) {
            Some(RedisValue::Stream(stream)) => stream.last().map(|entry| entry.id.to_string()),
            _ => None,
        }
    }

    fn add_blocked_stream_client(&mut self, client: BlockedStreamClient) {
        for (key, _) in &client.streams {
            self.blocked_stream_clients
                .entry(key.clone())
                .or_default()
                .push(client.clone());
        }
    }

    fn notify_blocked_stream_clients(&mut self, key: &str) {
        let mut clients_to_notify = Vec::new();
        let mut clients_to_remove = Vec::new();

        // First, collect clients and check for available data
        if let Some(clients) = self.blocked_stream_clients.get(key) {
            for client in clients {
                let streams_vec = client.streams.to_vec();
                match self.xread(streams_vec) {
                    Ok(result) if !result.is_empty() => {
                        clients_to_notify.push((client.response_sender.clone(), result));
                        clients_to_remove.push(client.client_id);
                    }
                    _ => {}
                }
            }
        }

        // Send responses
        for (sender, result) in clients_to_notify {
            let _ = sender.send(Some(result));
        }

        // Remove notified clients
        if !clients_to_remove.is_empty() {
            if let Some(clients) = self.blocked_stream_clients.get_mut(key) {
                clients.retain(|client| !clients_to_remove.contains(&client.client_id));
                if clients.is_empty() {
                    self.blocked_stream_clients.remove(key);
                }
            }
        }
    }

    fn generate_stream_client_id(&mut self) -> u64 {
        self.next_stream_client_id += 1;
        self.next_stream_client_id
    }

    fn cleanup_blocked_stream_client(&mut self, client_id: u64, streams: &[(String, String)]) {
        for (key, _) in streams {
            if let Some(clients) = self.blocked_stream_clients.get_mut(key) {
                clients.retain(|client| client.client_id != client_id);
                if clients.is_empty() {
                    self.blocked_stream_clients.remove(key);
                }
            }
        }
    }
}

impl Storage for MemoryStorage {}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}
