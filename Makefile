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
LAUNCH_AGENT := $(HOME)/Library/LaunchAgents/local.wtf.autostart.plist
UNAME_S := $(shell uname -s)
# Keep in sync with "identifier" in src-tauri/tauri.conf.json.
BUNDLE_ID := htol.wtf

.PHONY: dev build smoke install enable check clean npm-install signing-cert

npm-install:
	npm install

ifeq ($(UNAME_S),Darwin)
# Metal ships with macOS, so the dev build runs whisper on the GPU too.
dev: npm-install
	npm run tauri dev -- --features asr-metal
else
dev: npm-install
	npm run tauri dev
endif

ifeq ($(UNAME_S),Darwin)
# macOS ties the microphone and Accessibility grants to the code signature.
# An ad-hoc signature ("-") changes with every build and loses them; a
# self-signed certificate in the keychain keeps them (README, macOS).
SIGN_CERT := wtf-dev
SIGN_IDENTITY ?= $(shell security find-identity -p codesigning 2>/dev/null | grep -q '"$(SIGN_CERT)"' && echo $(SIGN_CERT) || echo -)

# Creates the signing certificate in the login keychain unless it is there.
# The grants given to earlier ad-hoc builds are reset once: they would
# shadow those of the newly signed app. A failure leaves the build ad-hoc.
signing-cert:
	-@if security find-identity -p codesigning | grep -q '"$(SIGN_CERT)"'; then exit 0; fi; \
	set -e; \
	tmp=$$(mktemp -d); \
	trap 'rm -f "$$tmp/cert.conf" "$$tmp/key.pem" "$$tmp/cert.pem" "$$tmp/cert.p12"; rmdir "$$tmp"' EXIT; \
	printf '%s\n' '[req]' 'distinguished_name = dn' 'x509_extensions = ext' 'prompt = no' \
		'[dn]' 'CN = $(SIGN_CERT)' \
		'[ext]' 'basicConstraints = critical,CA:false' 'keyUsage = critical,digitalSignature' \
		'extendedKeyUsage = critical,codeSigning' > "$$tmp/cert.conf"; \
	/usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$$tmp/cert.conf" \
		-keyout "$$tmp/key.pem" -out "$$tmp/cert.pem"; \
	/usr/bin/openssl pkcs12 -export -inkey "$$tmp/key.pem" -in "$$tmp/cert.pem" \
		-name $(SIGN_CERT) -out "$$tmp/cert.p12" -passout pass:$(SIGN_CERT); \
	security import "$$tmp/cert.p12" -k $(HOME)/Library/Keychains/login.keychain-db \
		-P $(SIGN_CERT) -T /usr/bin/codesign; \
	tccutil reset Accessibility $(BUNDLE_ID); \
	tccutil reset Microphone $(BUNDLE_ID); \
	echo "Created the $(SIGN_CERT) signing certificate; grant the permissions once more."

# Production build: wtf.app with the frontend dist embedded and ASR on the
# GPU via Metal.
build: npm-install signing-cert
	npm run tauri build -- --features prod,asr-metal \
		--config '{"bundle":{"macOS":{"signingIdentity":"$(SIGN_IDENTITY)"}}}'
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
	@echo "Installed $(INSTALL_APP). Start it (now + on login) with: make enable"
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

ifeq ($(UNAME_S),Darwin)
# A LaunchAgent that opens the installed bundle at login; loading it also
# starts the app now.
enable:
	mkdir -p $(dir $(LAUNCH_AGENT))
	sed 's|@APP@|$(INSTALL_APP)|' assets/wtf.launchagent.plist > $(LAUNCH_AGENT)
	-launchctl bootout gui/$$(id -u) $(LAUNCH_AGENT) 2>/dev/null
	launchctl bootstrap gui/$$(id -u) $(LAUNCH_AGENT)
else
enable:
	systemctl --user enable --now $(UNIT_NAME)
endif

clean:
	cargo clean --manifest-path src-tauri/Cargo.toml
	rm -rf dist node_modules/.vite
