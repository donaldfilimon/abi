#!/bin/bash
# Deployment script for WDBX-AI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="wdbx-ai"
SERVICE_NAME="wdbx-ai.service"
INSTALL_DIR="/opt/wdbx-ai"
CONFIG_DIR="/etc/wdbx-ai"
DATA_DIR="/var/lib/wdbx-ai"
LOG_DIR="/var/log/wdbx-ai"
USER="wdbx"
GROUP="wdbx"

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_msg "$RED" "This script must be run as root"
        exit 1
    fi
}

# Create system user
create_user() {
    print_msg "$BLUE" "Creating system user..."
    
    if ! id "$USER" &>/dev/null; then
        useradd --system --shell /bin/false --home-dir "$DATA_DIR" --create-home "$USER"
        print_msg "$GREEN" "Created user: $USER"
    else
        print_msg "$YELLOW" "User $USER already exists"
    fi
}

# Create directories
create_directories() {
    print_msg "$BLUE" "Creating directories..."
    
    # Create directories
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$LOG_DIR"
    
    # Set permissions
    chown -R "$USER:$GROUP" "$DATA_DIR"
    chown -R "$USER:$GROUP" "$LOG_DIR"
    chmod 750 "$DATA_DIR"
    chmod 750 "$LOG_DIR"
    
    print_msg "$GREEN" "Directories created"
}

# Install binaries
install_binaries() {
    local source_dir=${1:-"zig-out/bin"}
    
    print_msg "$BLUE" "Installing binaries..."
    
    if [[ ! -d "$source_dir" ]]; then
        print_msg "$RED" "Error: Source directory $source_dir not found"
        print_msg "$YELLOW" "Please build the project first: ./scripts/build.sh build release"
        exit 1
    fi
    
    # Copy binaries
    cp -f "$source_dir/wdbx" "$INSTALL_DIR/" || true
    cp -f "$source_dir/wdbx-cli" "$INSTALL_DIR/" || true
    
    # Make executable
    chmod +x "$INSTALL_DIR/wdbx" 2>/dev/null || true
    chmod +x "$INSTALL_DIR/wdbx-cli" 2>/dev/null || true
    
    # Create symlinks
    ln -sf "$INSTALL_DIR/wdbx" "/usr/local/bin/wdbx"
    ln -sf "$INSTALL_DIR/wdbx-cli" "/usr/local/bin/wdbx-cli"
    
    print_msg "$GREEN" "Binaries installed"
}

# Install configuration
install_config() {
    print_msg "$BLUE" "Installing configuration..."
    
    # Copy default configuration
    if [[ -f "config/default.toml" ]]; then
        if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
            cp "config/default.toml" "$CONFIG_DIR/config.toml"
            print_msg "$GREEN" "Installed default configuration"
        else
            cp "config/default.toml" "$CONFIG_DIR/config.toml.new"
            print_msg "$YELLOW" "New configuration saved as config.toml.new"
        fi
    fi
    
    # Set permissions
    chmod 640 "$CONFIG_DIR"/*.toml 2>/dev/null || true
    chown root:"$GROUP" "$CONFIG_DIR"/*.toml 2>/dev/null || true
}

# Create systemd service
create_systemd_service() {
    print_msg "$BLUE" "Creating systemd service..."
    
    cat > "/etc/systemd/system/$SERVICE_NAME" << EOF
[Unit]
Description=WDBX-AI Vector Database
Documentation=https://github.com/wdbx/wdbx-ai
After=network.target

[Service]
Type=simple
User=$USER
Group=$GROUP
WorkingDirectory=$DATA_DIR
ExecStart=$INSTALL_DIR/wdbx serve --config $CONFIG_DIR/config.toml
ExecReload=/bin/kill -USR1 \$MAINPID
Restart=on-failure
RestartSec=5
StandardOutput=append:$LOG_DIR/wdbx.log
StandardError=append:$LOG_DIR/wdbx-error.log

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA_DIR $LOG_DIR
ReadOnlyPaths=$CONFIG_DIR
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictRealtime=true
RestrictNamespaces=true
LockPersonality=true
MemoryDenyWriteExecute=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
SystemCallFilter=@system-service
SystemCallErrorNumber=EPERM

# Resource limits
LimitNOFILE=65536
LimitNPROC=512

# Environment
Environment="WDBX_DATABASE_PATH=$DATA_DIR/wdbx.db"
Environment="WDBX_LOG_LEVEL=info"

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
    
    print_msg "$GREEN" "Systemd service created"
}

# Setup log rotation
setup_logrotate() {
    print_msg "$BLUE" "Setting up log rotation..."
    
    cat > "/etc/logrotate.d/wdbx-ai" << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 $USER $GROUP
    sharedscripts
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF

    print_msg "$GREEN" "Log rotation configured"
}

# Start service
start_service() {
    print_msg "$BLUE" "Starting service..."
    
    systemctl enable "$SERVICE_NAME"
    systemctl start "$SERVICE_NAME"
    
    # Wait for service to start
    sleep 2
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        print_msg "$GREEN" "Service started successfully"
    else
        print_msg "$RED" "Failed to start service"
        print_msg "$YELLOW" "Check logs: journalctl -u $SERVICE_NAME -n 50"
        exit 1
    fi
}

# Stop service
stop_service() {
    print_msg "$BLUE" "Stopping service..."
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        systemctl stop "$SERVICE_NAME"
        print_msg "$GREEN" "Service stopped"
    else
        print_msg "$YELLOW" "Service is not running"
    fi
}

# Show service status
show_status() {
    print_msg "$BLUE" "Service status:"
    systemctl status "$SERVICE_NAME" --no-pager || true
    
    echo
    print_msg "$BLUE" "Recent logs:"
    journalctl -u "$SERVICE_NAME" -n 20 --no-pager || true
}

# Uninstall
uninstall() {
    print_msg "$YELLOW" "Uninstalling WDBX-AI..."
    
    # Stop service
    stop_service
    
    # Disable service
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    
    # Remove files
    rm -f "/etc/systemd/system/$SERVICE_NAME"
    rm -f "/etc/logrotate.d/wdbx-ai"
    rm -f "/usr/local/bin/wdbx"
    rm -f "/usr/local/bin/wdbx-cli"
    rm -rf "$INSTALL_DIR"
    
    # Reload systemd
    systemctl daemon-reload
    
    print_msg "$GREEN" "Uninstall complete"
    print_msg "$YELLOW" "Note: Configuration, data, and logs were preserved in:"
    print_msg "$YELLOW" "  - $CONFIG_DIR"
    print_msg "$YELLOW" "  - $DATA_DIR"
    print_msg "$YELLOW" "  - $LOG_DIR"
}

# Backup data
backup() {
    local backup_dir=${1:-"/tmp/wdbx-backup-$(date +%Y%m%d-%H%M%S)"}
    
    print_msg "$BLUE" "Creating backup in $backup_dir..."
    
    mkdir -p "$backup_dir"
    
    # Stop service during backup
    local was_running=false
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        was_running=true
        stop_service
    fi
    
    # Backup data
    cp -r "$DATA_DIR" "$backup_dir/" 2>/dev/null || true
    cp -r "$CONFIG_DIR" "$backup_dir/" 2>/dev/null || true
    
    # Create backup info
    cat > "$backup_dir/backup-info.txt" << EOF
WDBX-AI Backup
Date: $(date)
Version: $(wdbx --version 2>/dev/null || echo "unknown")
Data directory: $DATA_DIR
Config directory: $CONFIG_DIR
EOF

    # Create archive
    tar -czf "$backup_dir.tar.gz" -C "$(dirname "$backup_dir")" "$(basename "$backup_dir")"
    rm -rf "$backup_dir"
    
    # Restart service if it was running
    if [[ "$was_running" == "true" ]]; then
        start_service
    fi
    
    print_msg "$GREEN" "Backup created: $backup_dir.tar.gz"
}

# Restore from backup
restore() {
    local backup_file=$1
    
    if [[ -z "$backup_file" ]]; then
        print_msg "$RED" "Error: Backup file not specified"
        exit 1
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        print_msg "$RED" "Error: Backup file not found: $backup_file"
        exit 1
    fi
    
    print_msg "$BLUE" "Restoring from $backup_file..."
    
    # Stop service
    stop_service
    
    # Extract backup
    local temp_dir="/tmp/wdbx-restore-$$"
    mkdir -p "$temp_dir"
    tar -xzf "$backup_file" -C "$temp_dir"
    
    # Find backup directory
    local backup_dir=$(find "$temp_dir" -name "backup-info.txt" -exec dirname {} \; | head -1)
    
    if [[ -z "$backup_dir" ]]; then
        print_msg "$RED" "Error: Invalid backup file"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Restore data
    cp -r "$backup_dir/$(basename "$DATA_DIR")"/* "$DATA_DIR/" 2>/dev/null || true
    cp -r "$backup_dir/$(basename "$CONFIG_DIR")"/* "$CONFIG_DIR/" 2>/dev/null || true
    
    # Set permissions
    chown -R "$USER:$GROUP" "$DATA_DIR"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    # Start service
    start_service
    
    print_msg "$GREEN" "Restore complete"
}

# Show help
show_help() {
    cat << EOF
WDBX-AI Deployment Script

Usage: $0 [command] [options]

Commands:
    install          Install WDBX-AI
    uninstall        Uninstall WDBX-AI
    start            Start the service
    stop             Stop the service
    restart          Restart the service
    status           Show service status
    backup [dir]     Create backup
    restore <file>   Restore from backup
    help             Show this help message

Examples:
    $0 install                    # Install WDBX-AI
    $0 status                     # Check service status
    $0 backup                     # Create backup
    $0 restore backup.tar.gz      # Restore from backup

Note: This script must be run as root.

EOF
}

# Main script
main() {
    local command=${1:-help}
    shift || true
    
    case $command in
        install)
            check_root
            create_user
            create_directories
            install_binaries "$@"
            install_config
            create_systemd_service
            setup_logrotate
            start_service
            show_status
            print_msg "$GREEN" "Installation complete!"
            ;;
        uninstall)
            check_root
            uninstall
            ;;
        start)
            check_root
            start_service
            ;;
        stop)
            check_root
            stop_service
            ;;
        restart)
            check_root
            stop_service
            start_service
            ;;
        status)
            show_status
            ;;
        backup)
            check_root
            backup "$@"
            ;;
        restore)
            check_root
            restore "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_msg "$RED" "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"