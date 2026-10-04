BIN_NAME := wtf
INSTALL_BIN := $(HOME)/.local/bin/$(BIN_NAME)
# The "app-" prefix is required: xdg-desktop-portal derives the portal
# app id for unsandboxed processes from the systemd user unit name
# (unit must be app-<appid>[@...].service), then loads <appid>.desktop.
# Without it GlobalShortcuts fails with "An app id is required".
UNIT_NAME := app-$(BIN_NAME).service
OLD_UNIT_NAME := $(BIN_NAME).service
UNIT_DIR := $(HOME)/.config/systemd/user
DESKTOP_DIR := $(HOME)/.local/share/applications
ICON_DIR := $(HOME)/.local/share/icons/hicolor/128x128/apps
RELEASE_BIN := src-tauri/target/release/$(BIN_NAME)
# macOS: the app is built and installed as a bundle (the microphone
# permission needs its Info.plist).
APP_BUNDLE := src-tauri/target/release/bundle/macos/$(BIN_NAME).app
INSTALL_APP := $(HOME)/Applications/$(BIN_NAME).app
UNAME_S := $(shell uname -s)

.PHONY: dev build smoke install enable check clean npm-install

npm-install:
	npm install

dev: npm-install
	npm run tauri dev

ifeq ($(UNAME_S),Darwin)
# Production build: wtf.app with the frontend dist embedded and ASR on the
# GPU via Metal.
build: npm-install
	npm run tauri build -- --features prod,asr-metal
else
# Production build: embed the frontend dist into the binary and run ASR on
# the GPU via Vulkan — any vendor driver (RADV, NVIDIA proprietary, ...);
# runtime device pick via settings.gpu_device.
build: npm-install
	npm run build
	cargo build --release --manifest-path src-tauri/Cargo.toml --features prod,asr-vulkan
endif

# Validates that cuda + vulkan backends link into one binary (DESIGN.md risk #1).
smoke: npm-install
	npm run build
	cargo build --manifest-path src-tauri/Cargo.toml --features asr-cuda,asr-vulkan

check:
	cargo check --manifest-path src-tauri/Cargo.toml

ifeq ($(UNAME_S),Darwin)
install: build
	mkdir -p $(HOME)/Applications
	ditto $(APP_BUNDLE) $(INSTALL_APP)
	@echo "Installed $(INSTALL_APP)"
else
install: build
	install -Dm755 $(RELEASE_BIN) $(INSTALL_BIN)
	install -Dm644 assets/$(UNIT_NAME) $(UNIT_DIR)/$(UNIT_NAME)
	install -Dm644 assets/wtf.desktop $(DESKTOP_DIR)/wtf.desktop
	install -Dm644 src-tauri/icons/icon.png $(ICON_DIR)/wtf.png
	# One-time migration from the pre-portal unit name.
	-systemctl --user disable --now $(OLD_UNIT_NAME) 2>/dev/null
	-rm -f $(UNIT_DIR)/$(OLD_UNIT_NAME)
	systemctl --user daemon-reload
	@echo "Installed. Start it (now + on login) with: make enable"
endif

enable:
	systemctl --user enable --now $(UNIT_NAME)

clean:
	cargo clean --manifest-path src-tauri/Cargo.toml
	rm -rf dist node_modules/.vite
