package adapters

// NotificationPayload holds the formatted notification content
type NotificationPayload struct {
	Subject  string // used by email
	BodyText string // plain text
	BodyHTML string // HTML (email)
	Markdown string // Markdown (Telegram, Slack)
}

// ChannelAdapter is the interface all notification channels implement
type ChannelAdapter interface {
	Name() string
	Send(payload *NotificationPayload) error
}
