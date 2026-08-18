//! Dictation history in SQLite (text + language + timestamp, kept forever,
//! audio never stored). Only successful transcriptions are recorded.

use rusqlite::Connection;

use crate::app_id;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Entry {
	pub id: i64,
	pub text: String,
	pub language: String,
	pub created_at: i64, // unix seconds
}

pub fn open() -> Result<Connection, String> {
	let dir = app_id::data_dir();
	std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
	let conn = Connection::open(dir.join("history.db")).map_err(|e| e.to_string())?;
	conn.execute_batch(
		"CREATE TABLE IF NOT EXISTS history (
			id          INTEGER PRIMARY KEY,
			text        TEXT NOT NULL,
			language    TEXT NOT NULL,
			created_at  INTEGER NOT NULL
		);",
	)
	.map_err(|e| e.to_string())?;
	Ok(conn)
}

pub fn insert(conn: &Connection, text: &str, language: &str) -> Result<(), String> {
	conn.execute(
		"INSERT INTO history (text, language, created_at) VALUES (?1, ?2, unixepoch())",
		rusqlite::params![text, language],
	)
	.map_err(|e| e.to_string())?;
	Ok(())
}

pub fn list(conn: &Connection, limit: u32) -> Result<Vec<Entry>, String> {
	let mut stmt = conn
		.prepare("SELECT id, text, language, created_at FROM history ORDER BY id DESC LIMIT ?1")
		.map_err(|e| e.to_string())?;
	let rows = stmt
		.query_map([limit], |row| {
			Ok(Entry {
				id: row.get(0)?,
				text: row.get(1)?,
				language: row.get(2)?,
				created_at: row.get(3)?,
			})
		})
		.map_err(|e| e.to_string())?;
	rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_history(limit: Option<u32>) -> Result<Vec<Entry>, String> {
	let conn = open()?;
	list(&conn, limit.unwrap_or(100))
}

#[tauri::command]
pub fn delete_history(ids: Vec<i64>) -> Result<(), String> {
	let conn = open()?;
	// Chunk to stay well below SQLite's host-parameter limit.
	for chunk in ids.chunks(500) {
		let placeholders = vec!["?"; chunk.len()].join(",");
		let sql = format!("DELETE FROM history WHERE id IN ({placeholders})");
		conn.execute(&sql, rusqlite::params_from_iter(chunk))
			.map_err(|e| e.to_string())?;
	}
	Ok(())
}
