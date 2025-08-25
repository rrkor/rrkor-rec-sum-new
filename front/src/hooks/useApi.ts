import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, GigaChatSettings } from '../lib/api';
import { WebSocketClient, TranscriptMessage, LevelsMessage } from '../lib/websocket';
import { useToast } from './use-toast';

export function useApi() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [wsClient, setWsClient] = useState<WebSocketClient | null>(null);

  // Devices
  const devicesQuery = useQuery({
    queryKey: ['devices'],
    queryFn: api.getDevices,
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  const selectMicrophoneMutation = useMutation({
    mutationFn: api.selectMicrophone,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['devices'] });
      queryClient.invalidateQueries({ queryKey: ['state'] });
      toast({
        title: "Микрофон изменен",
        description: "Настройки микрофона обновлены",
      });
    },
    onError: (error) => {
      toast({
        title: "Ошибка",
        description: `Не удалось изменить микрофон: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  // State
  const stateQuery = useQuery({
    queryKey: ['state'],
    queryFn: api.getState,
    refetchInterval: 1000, // Refresh every second
  });

  // Recording
  const startRecordingMutation = useMutation({
    mutationFn: api.startRecording,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['state'] });
      toast({
        title: "Запись начата",
        description: "Запись аудио запущена",
      });
    },
    onError: (error) => {
      toast({
        title: "Ошибка",
        description: `Не удалось начать запись: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  const pauseRecordingMutation = useMutation({
    mutationFn: api.pauseRecording,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['state'] });
      toast({
        title: "Запись приостановлена",
        description: "Запись поставлена на паузу",
      });
    },
    onError: (error) => {
      toast({
        title: "Ошибка",
        description: `Не удалось приостановить запись: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  const resumeRecordingMutation = useMutation({
    mutationFn: api.resumeRecording,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['state'] });
      toast({
        title: "Запись возобновлена",
        description: "Запись продолжена",
      });
    },
    onError: (error) => {
      toast({
        title: "Ошибка",
        description: `Не удалось возобновить запись: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  const stopRecordingMutation = useMutation({
    mutationFn: api.stopRecording,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['state'] });
      queryClient.invalidateQueries({ queryKey: ['transcript'] });
      if (data.ok) {
        toast({
          title: "Запись остановлена",
          description: "Запись завершена",
        });
      } else {
        toast({
          title: "Предупреждение",
          description: data.error || "Запись остановлена с предупреждениями",
          variant: "destructive",
        });
      }
    },
    onError: (error) => {
      toast({
        title: "Ошибка",
        description: `Не удалось остановить запись: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  // Settings
  const updateSettingsMutation = useMutation({
    mutationFn: api.updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['state'] });
      toast({
        title: "Настройки обновлены",
        description: "Параметры записи изменены",
      });
    },
    onError: (error) => {
      toast({
        title: "Ошибка",
        description: `Не удалось обновить настройки: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  // Transcript
  const transcriptQuery = useQuery({
    queryKey: ['transcript'],
    queryFn: api.getTranscript,
    refetchInterval: (data) => {
      // Only refetch if recording is active
      const state = queryClient.getQueryData(['state']) as any;
      return state?.recording ? 2000 : false;
    },
  });

  const saveTranscriptMutation = useMutation({
    mutationFn: api.saveTranscript,
    onSuccess: (data) => {
      toast({
        title: "Транскрипт сохранен",
        description: `Файл сохранен: ${data.path}`,
      });
    },
    onError: (error) => {
      toast({
        title: "Ошибка",
        description: `Не удалось сохранить транскрипт: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  // Summary
  const summarizeMutation = useMutation({
    mutationFn: api.summarize,
    onSuccess: () => {
      toast({
        title: "Саммари создано",
        description: "Краткое содержание готово",
      });
    },
    onError: (error) => {
      toast({
        title: "Ошибка",
        description: `Не удалось создать саммари: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  // GigaChat Settings
  const gigachatSettingsQuery = useQuery({
    queryKey: ['gigachat-settings'],
    queryFn: api.getGigaChatSettings,
  });

  const saveGigaChatSettingsMutation = useMutation({
    mutationFn: api.saveGigaChatSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['gigachat-settings'] });
      toast({
        title: "Настройки сохранены",
        description: "Конфигурация GigaChat обновлена",
      });
    },
    onError: (error) => {
      toast({
        title: "Ошибка",
        description: `Не удалось сохранить настройки: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  // WebSocket setup
  const setupWebSocket = useCallback((onTranscriptMessage?: (message: TranscriptMessage) => void, onLevelsMessage?: (message: LevelsMessage) => void) => {
    // Если WebSocket уже подключен и работает, не переподключаемся
    if (wsClient && wsClient.isConnected()) {
      return wsClient;
    }

    // Отключаем старый клиент если есть
    if (wsClient) {
      wsClient.disconnect();
    }

    const client = new WebSocketClient(
      onTranscriptMessage,
      onLevelsMessage,
      (error) => {
        console.error('WebSocket error:', error);
        // Не показываем toast для каждой ошибки, только для критических
        if (error.type === 'error') {
          toast({
            title: "Ошибка соединения",
            description: "Проблема с WebSocket соединением",
            variant: "destructive",
          });
        }
      }
    );

    client.connect();
    setWsClient(client);

    return client;
  }, [wsClient, toast]);

  useEffect(() => {
    return () => {
      if (wsClient) {
        wsClient.disconnect();
      }
    };
  }, [wsClient]);

  return {
    // Queries
    devices: devicesQuery,
    state: stateQuery,
    transcript: transcriptQuery,
    
    // Mutations
    selectMicrophone: selectMicrophoneMutation,
    startRecording: startRecordingMutation,
    pauseRecording: pauseRecordingMutation,
    resumeRecording: resumeRecordingMutation,
    stopRecording: stopRecordingMutation,
    updateSettings: updateSettingsMutation,
    saveTranscript: saveTranscriptMutation,
    summarize: summarizeMutation,
    gigachatSettings: gigachatSettingsQuery,
    saveGigaChatSettings: saveGigaChatSettingsMutation,
    
    // WebSocket
    setupWebSocket,
    wsClient,
  };
}
