const API_BASE = 'http://127.0.0.1:8000/api';

export interface Device {
  name: string;
}

export interface DevicesResponse {
  microphones: string[];
  selected: string;
  blackhole_present: boolean;
  blackhole_name: string;
}

export interface StateResponse {
  recording: boolean;
  paused: boolean;
  mic_gain: number;
  blackhole_gain: number;
  mic_silence_threshold: number;
  blackhole_silence_threshold: number;
  mic_db: number;
  blackhole_db: number;
  selected_mic: string;
}

export interface SettingsPatch {
  mic_gain?: number;
  blackhole_gain?: number;
  mic_silence_threshold?: number;
  blackhole_silence_threshold?: number;
}

export interface TranscriptResponse {
  text: string;
}

export interface SummaryResponse {
  summary: string;
}

export interface SaveResponse {
  ok: boolean;
  path: string;
  bytes: number;
}

export interface GigaChatSettings {
  credentials: string;
  scope: string;
  model: string;
}

class ApiClient {
  private request = async <T>(endpoint: string, options?: RequestInit): Promise<T> => {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  };

  // Devices
  getDevices = async (): Promise<DevicesResponse> => {
    return this.request<DevicesResponse>('/devices');
  };

  selectMicrophone = async (name: string): Promise<{ ok: boolean; selected: string }> => {
    return this.request<{ ok: boolean; selected: string }>('/microphone', {
      method: 'POST',
      body: JSON.stringify({ name }),
    });
  };

  // State
  getState = async (): Promise<StateResponse> => {
    return this.request<StateResponse>('/state');
  };

  // Recording
  startRecording = async (): Promise<{ ok: boolean; status?: string }> => {
    return this.request<{ ok: boolean; status?: string }>('/record/start', {
      method: 'POST',
    });
  };

  pauseRecording = async (): Promise<{ ok: boolean }> => {
    return this.request<{ ok: boolean }>('/record/pause', {
      method: 'POST',
    });
  };

  resumeRecording = async (): Promise<{ ok: boolean }> => {
    return this.request<{ ok: boolean }>('/record/resume', {
      method: 'POST',
    });
  };

  stopRecording = async (): Promise<{ ok: boolean }> => {
    return this.request<{ ok: boolean }>('/record/stop', {
      method: 'POST',
    });
  };

  // Settings
  updateSettings = async (settings: SettingsPatch): Promise<{ ok: boolean; settings: any }> => {
    return this.request<{ ok: boolean; settings: any }>('/settings', {
      method: 'PATCH',
      body: JSON.stringify(settings),
    });
  };

  // Transcript
  getTranscript = async (): Promise<TranscriptResponse> => {
    return this.request<TranscriptResponse>('/transcript');
  };

  saveTranscript = async (): Promise<SaveResponse> => {
    return this.request<SaveResponse>('/transcript/save', {
      method: 'POST',
    });
  };

  // Summary
  summarize = async (text?: string): Promise<SummaryResponse> => {
    return this.request<SummaryResponse>('/summarize', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  };

  // Health
  health = async (): Promise<{ ok: boolean }> => {
    return this.request<{ ok: boolean }>('/health');
  };

  // GigaChat Settings
  getGigaChatSettings = async (): Promise<GigaChatSettings> => {
    return this.request<GigaChatSettings>('/gigachat-settings');
  };

  saveGigaChatSettings = async (settings: GigaChatSettings): Promise<{ ok: boolean; message: string }> => {
    return this.request<{ ok: boolean; message: string }>('/gigachat-settings', {
      method: 'POST',
      body: JSON.stringify(settings),
    });
  };
}

export const api = new ApiClient();
