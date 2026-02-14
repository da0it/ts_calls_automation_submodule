package services

import (
	"log"
	"sync"

	"notification_sender/internal/adapters"
	callprocessingv1 "notification_sender/internal/gen"
	"notification_sender/internal/templates"
)

type NotificationService struct {
	channels []adapters.ChannelAdapter
}

func NewNotificationService(channels []adapters.ChannelAdapter) *NotificationService {
	return &NotificationService{channels: channels}
}

type ChannelResult struct {
	ChannelName string
	Success     bool
	Error       string
}

func (s *NotificationService) SendNotification(req *callprocessingv1.SendNotificationRequest) (bool, []ChannelResult) {
	payload := templates.FormatNotification(req)

	results := make([]ChannelResult, len(s.channels))
	var wg sync.WaitGroup

	for i, ch := range s.channels {
		wg.Add(1)
		go func(idx int, channel adapters.ChannelAdapter) {
			defer wg.Done()
			log.Printf("Sending notification via %s...", channel.Name())

			err := channel.Send(payload)
			if err != nil {
				log.Printf("Notification via %s failed: %v", channel.Name(), err)
				results[idx] = ChannelResult{
					ChannelName: channel.Name(),
					Success:     false,
					Error:       err.Error(),
				}
			} else {
				log.Printf("Notification via %s sent successfully", channel.Name())
				results[idx] = ChannelResult{
					ChannelName: channel.Name(),
					Success:     true,
				}
			}
		}(i, ch)
	}

	wg.Wait()

	anySuccess := false
	for _, r := range results {
		if r.Success {
			anySuccess = true
			break
		}
	}

	return anySuccess, results
}
