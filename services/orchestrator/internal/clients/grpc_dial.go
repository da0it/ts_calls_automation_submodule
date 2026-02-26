package clients

import (
	"crypto/tls"
	"fmt"
	"os"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

func grpcConnForService(addr string, envPrefix string) (*grpc.ClientConn, error) {
	transportCreds, err := grpcTransportCredentials(envPrefix)
	if err != nil {
		return nil, fmt.Errorf("build transport credentials for %s: %w", envPrefix, err)
	}
	conn, err := grpc.NewClient(
		addr,
		grpc.WithTransportCredentials(transportCreds),
	)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

func grpcTransportCredentials(envPrefix string) (credentials.TransportCredentials, error) {
	tlsEnabled := envBool(
		envPrefix+"_TLS_ENABLED",
		envBool("GRPC_TLS_ENABLED", false),
	)
	if !tlsEnabled {
		return insecure.NewCredentials(), nil
	}

	caFile := strings.TrimSpace(os.Getenv(envPrefix + "_TLS_CA_FILE"))
	if caFile == "" {
		caFile = strings.TrimSpace(os.Getenv("GRPC_TLS_CA_FILE"))
	}
	serverName := strings.TrimSpace(os.Getenv(envPrefix + "_TLS_SERVER_NAME"))
	if serverName == "" {
		serverName = strings.TrimSpace(os.Getenv("GRPC_TLS_SERVER_NAME"))
	}

	if caFile != "" {
		creds, err := credentials.NewClientTLSFromFile(caFile, serverName)
		if err != nil {
			return nil, err
		}
		return creds, nil
	}

	cfg := &tls.Config{
		MinVersion: tls.VersionTLS12,
		ServerName: serverName,
	}
	return credentials.NewTLS(cfg), nil
}

func envBool(name string, def bool) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	if raw == "" {
		return def
	}
	switch raw {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return def
	}
}
