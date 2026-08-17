BIN_NAME := wtf
INSTALL_BIN := $(HOME)/.local/bin/$(BIN_NAME)
UNIT_NAME := $(BIN_NAME).service
UNIT_DIR := $(HOME)/.config/systemd/user
RELEASE_BIN := src-tauri/target/release/$(BIN_NAME)

.PHONY: dev build smoke install enable check clean npm-install

npm-install:
	npm install

dev: npm-install
	npm run tauri dev

build: npm-install
	npm run build
	cargo build --release --manifest-path src-tauri/Cargo.toml --features prod

# Validates that cuda + vulkan backends link into one binary (DESIGN.md risk #1).
smoke: npm-install
	npm run build
	cargo build --manifest-path src-tauri/Cargo.toml --features asr-cuda,asr-vulkan

check:
	cargo check --manifest-path src-tauri/Cargo.toml

install: build
	install -Dm755 $(RELEASE_BIN) $(INSTALL_BIN)
	install -Dm644 assets/$(UNIT_NAME) $(UNIT_DIR)/$(UNIT_NAME)
	systemctl --user daemon-reload
	@echo "Installed. Start it (now + on login) with: make enable"

enable:
	systemctl --user enable --now $(UNIT_NAME)

clean:
	cargo clean --manifest-path src-tauri/Cargo.toml
	rm -rf dist node_modules/.vite
