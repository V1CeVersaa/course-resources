pub trait BasicStorage: Send + Sync {
    fn set(&mut self, key: String, value: String) -> Result<(), String>;
    fn get(&self, key: &str) -> Option<String>;
    fn expire(&mut self, key: &str, seconds: u64) -> Result<bool, String>;
    fn get_type(&self, key: &str) -> String;
    fn incr(&mut self, key: &str) -> Result<i64, String>;
}
