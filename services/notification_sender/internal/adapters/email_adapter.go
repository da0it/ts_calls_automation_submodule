package adapters

import (
	"fmt"
	"net/smtp"
	"strings"
)

type EmailAdapter struct {
	host     string
	port     string
	user     string
	password string
	from     string
	to       []string
}

func NewEmailAdapter(host, port, user, password, from, to string) *EmailAdapter {
	var recipients []string
	for _, r := range strings.Split(to, ",") {
		r = strings.TrimSpace(r)
		if r != "" {
			recipients = append(recipients, r)
		}
	}
	return &EmailAdapter{
		host: host, port: port,
		user: user, password: password,
		from: from, to: recipients,
	}
}

func (a *EmailAdapter) Name() string { return "email" }

func (a *EmailAdapter) Send(payload *NotificationPayload) error {
	if len(a.to) == 0 {
		return fmt.Errorf("no email recipients configured")
	}

	addr := fmt.Sprintf("%s:%s", a.host, a.port)

	headers := fmt.Sprintf(
		"From: %s\r\nTo: %s\r\nSubject: %s\r\nMIME-Version: 1.0\r\nContent-Type: text/html; charset=\"UTF-8\"\r\n\r\n",
		a.from, strings.Join(a.to, ","), payload.Subject,
	)
	body := headers + payload.BodyHTML

	var auth smtp.Auth
	if a.user != "" {
		auth = smtp.PlainAuth("", a.user, a.password, a.host)
	}

	return smtp.SendMail(addr, auth, a.from, a.to, []byte(body))
}
