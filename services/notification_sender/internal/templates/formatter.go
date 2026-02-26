package templates

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"notification_sender/internal/adapters"
	callprocessingv1 "notification_sender/internal/gen"
)

var (
	emailPattern      = regexp.MustCompile(`(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b`)
	phonePattern      = regexp.MustCompile(`\+?\d[\d\-\s\(\)]{8,}\d`)
	longNumberPattern = regexp.MustCompile(`\b\d{5,}\b`)
)

// FormatNotification builds the notification payload from pipeline results
func FormatNotification(req *callprocessingv1.SendNotificationRequest) *adapters.NotificationPayload {
	includePII := envBool("NOTIFICATION_INCLUDE_PII", false)
	redactSummary := envBool("NOTIFICATION_REDACT_SUMMARY", true)

	ticket := req.GetTicket()
	routing := req.GetRouting()
	entities := req.GetEntities()
	transcript := req.GetTranscript()

	priority := routing.GetPriority()
	intent := routing.GetIntentId()
	group := routing.GetSuggestedGroup()
	ticketID := ticket.GetTicketId()
	ticketURL := ticket.GetUrl()
	callID := req.GetCallId()

	summary := buildCallSummary(transcript)

	persons := entityValues(entities.GetPersons())
	phones := entityValues(entities.GetPhones())
	emails := entityValues(entities.GetEmails())
	orderIDs := entityValues(entities.GetOrderIds())

	emoji := priorityEmoji(priority)
	if redactSummary {
		summary = redactSensitiveText(summary, persons, phones, emails, orderIDs)
	}

	// --- Subject ---
	subject := fmt.Sprintf("%s [%s] Тикет %s — %s",
		emoji, strings.ToUpper(priority), ticketID, intent)

	// --- Plain text ---
	var text strings.Builder
	fmt.Fprintf(&text, "Новый тикет создан\n")
	fmt.Fprintf(&text, "==================\n")
	fmt.Fprintf(&text, "Тикет ID:   %s\n", ticketID)
	fmt.Fprintf(&text, "URL:        %s\n", ticketURL)
	fmt.Fprintf(&text, "Приоритет:  %s\n", priority)
	fmt.Fprintf(&text, "Интент:     %s\n", intent)
	fmt.Fprintf(&text, "Группа:     %s\n", group)
	fmt.Fprintf(&text, "Call ID:    %s\n\n", callID)
	if summary != "" {
		fmt.Fprintf(&text, "Содержание звонка:\n%s\n\n", summary)
	}
	if includePII {
		appendList(&text, "Клиенты", persons)
		appendList(&text, "Телефоны", phones)
		appendList(&text, "Email", emails)
		appendList(&text, "Заказы", orderIDs)
	} else {
		appendCount(&text, "Клиенты", len(persons))
		appendCount(&text, "Телефоны", len(phones))
		appendCount(&text, "Email", len(emails))
		appendCount(&text, "Заказы", len(orderIDs))
	}

	// --- Markdown (Telegram / Slack) ---
	var md strings.Builder
	fmt.Fprintf(&md, "%s *Новый тикет создан*\n\n", emoji)
	fmt.Fprintf(&md, "*Тикет:* `%s`\n", ticketID)
	if ticketURL != "" {
		fmt.Fprintf(&md, "*URL:* %s\n", ticketURL)
	}
	fmt.Fprintf(&md, "*Приоритет:* %s\n", priority)
	fmt.Fprintf(&md, "*Интент:* %s\n", intent)
	fmt.Fprintf(&md, "*Группа:* %s\n", group)
	fmt.Fprintf(&md, "*Call ID:* `%s`\n", callID)
	if summary != "" {
		fmt.Fprintf(&md, "\n*Содержание:*\n%s\n", summary)
	}
	if includePII {
		appendListMd(&md, "Клиенты", persons)
		appendListMd(&md, "Телефоны", phones)
		appendListMd(&md, "Email", emails)
		appendListMd(&md, "Заказы", orderIDs)
	} else {
		appendCountMd(&md, "Клиенты", len(persons))
		appendCountMd(&md, "Телефоны", len(phones))
		appendCountMd(&md, "Email", len(emails))
		appendCountMd(&md, "Заказы", len(orderIDs))
	}

	// --- HTML (email) ---
	var html strings.Builder
	fmt.Fprintf(&html, "<h2>%s Новый тикет создан</h2>", emoji)
	fmt.Fprintf(&html, "<table style=\"border-collapse:collapse;\">")
	htmlRow(&html, "Тикет ID", ticketID)
	if ticketURL != "" {
		htmlRow(&html, "URL", fmt.Sprintf("<a href=\"%s\">%s</a>", ticketURL, ticketURL))
	}
	htmlRow(&html, "Приоритет", priority)
	htmlRow(&html, "Интент", intent)
	htmlRow(&html, "Группа", group)
	htmlRow(&html, "Call ID", callID)
	fmt.Fprintf(&html, "</table>")
	if summary != "" {
		fmt.Fprintf(&html, "<h3>Содержание звонка</h3><pre>%s</pre>", summary)
	}
	if includePII {
		appendListHTML(&html, "Клиенты", persons)
		appendListHTML(&html, "Телефоны", phones)
		appendListHTML(&html, "Email", emails)
		appendListHTML(&html, "Заказы", orderIDs)
	} else {
		appendCountHTML(&html, "Клиенты", len(persons))
		appendCountHTML(&html, "Телефоны", len(phones))
		appendCountHTML(&html, "Email", len(emails))
		appendCountHTML(&html, "Заказы", len(orderIDs))
	}

	return &adapters.NotificationPayload{
		Subject:  subject,
		BodyText: text.String(),
		BodyHTML: html.String(),
		Markdown: md.String(),
	}
}

func buildCallSummary(transcript *callprocessingv1.Transcript) string {
	if transcript == nil || len(transcript.GetSegments()) == 0 {
		return ""
	}
	var sb strings.Builder
	totalLen := 0
	for _, seg := range transcript.GetSegments() {
		role := seg.GetRole()
		if role == "" {
			role = seg.GetSpeaker()
		}
		line := fmt.Sprintf("[%s]: %s", role, seg.GetText())
		if totalLen+len(line) > 500 {
			sb.WriteString("\n...")
			break
		}
		sb.WriteString(line)
		sb.WriteString("\n")
		totalLen += len(line)
	}
	return strings.TrimSpace(sb.String())
}

func entityValues(entities []*callprocessingv1.ExtractedEntity) []string {
	var vals []string
	for _, e := range entities {
		if v := e.GetValue(); v != "" {
			vals = append(vals, v)
		}
	}
	return vals
}

func redactSensitiveText(text string, persons, phones, emails, orderIDs []string) string {
	if text == "" {
		return text
	}
	out := text
	out = emailPattern.ReplaceAllString(out, "[EMAIL]")
	out = phonePattern.ReplaceAllString(out, "[PHONE]")
	out = longNumberPattern.ReplaceAllString(out, "[ID]")
	out = replaceValues(out, persons, "[PERSON]")
	out = replaceValues(out, phones, "[PHONE]")
	out = replaceValues(out, emails, "[EMAIL]")
	out = replaceValues(out, orderIDs, "[ORDER_ID]")
	return out
}

func replaceValues(text string, values []string, token string) string {
	out := text
	for _, raw := range values {
		value := strings.TrimSpace(raw)
		if len([]rune(value)) < 3 {
			continue
		}
		pattern := regexp.MustCompile(`(?i)` + regexp.QuoteMeta(value))
		out = pattern.ReplaceAllString(out, token)
	}
	return out
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

func priorityEmoji(p string) string {
	switch strings.ToLower(p) {
	case "critical":
		return "\U0001F534" // red circle
	case "high":
		return "\U0001F7E0" // orange circle
	case "medium":
		return "\U0001F7E1" // yellow circle
	default:
		return "\U0001F7E2" // green circle
	}
}

func appendList(sb *strings.Builder, label string, items []string) {
	if len(items) == 0 {
		return
	}
	fmt.Fprintf(sb, "%s: %s\n", label, strings.Join(items, ", "))
}

func appendCount(sb *strings.Builder, label string, count int) {
	if count <= 0 {
		return
	}
	fmt.Fprintf(sb, "%s: %d\n", label, count)
}

func appendListMd(sb *strings.Builder, label string, items []string) {
	if len(items) == 0 {
		return
	}
	fmt.Fprintf(sb, "*%s:* %s\n", label, strings.Join(items, ", "))
}

func appendCountMd(sb *strings.Builder, label string, count int) {
	if count <= 0 {
		return
	}
	fmt.Fprintf(sb, "*%s:* %d\n", label, count)
}

func appendListHTML(sb *strings.Builder, label string, items []string) {
	if len(items) == 0 {
		return
	}
	fmt.Fprintf(sb, "<p><strong>%s:</strong> %s</p>", label, strings.Join(items, ", "))
}

func appendCountHTML(sb *strings.Builder, label string, count int) {
	if count <= 0 {
		return
	}
	fmt.Fprintf(sb, "<p><strong>%s:</strong> %d</p>", label, count)
}

func htmlRow(sb *strings.Builder, label, value string) {
	fmt.Fprintf(sb, "<tr><td style=\"padding:4px 12px 4px 0;font-weight:bold;\">%s</td><td style=\"padding:4px 0;\">%s</td></tr>", label, value)
}
