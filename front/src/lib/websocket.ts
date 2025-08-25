export interface TranscriptMessage {
  type: 'line';
  text: string;
  ts: number;
  role: 'peer' | 'you' | 'raw';
}

export interface LevelsMessage {
  mic_db: number;
  blackhole_db: number;
}

export class WebSocketClient {
  private transcriptWs: WebSocket | null = null;
  private levelsWs: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private shouldReconnect = true;

  constructor(
    private onTranscriptMessage?: (message: TranscriptMessage) => void,
    private onLevelsMessage?: (message: LevelsMessage) => void,
    private onError?: (error: Event) => void
  ) {}

  connect() {
    if (this.isConnecting) return;
    this.isConnecting = true;
    this.shouldReconnect = true;
    this.connectTranscript();
    this.connectLevels();
  }

  private connectTranscript() {
    try {
      if (this.transcriptWs?.readyState === WebSocket.OPEN) {
        return;
      }
      
      this.transcriptWs = new WebSocket('ws://127.0.0.1:8000/ws/transcript');
      
      this.transcriptWs.onopen = () => {
        console.log('Transcript WebSocket connected');
        this.reconnectAttempts = 0;
        this.isConnecting = false;
      };

      this.transcriptWs.onmessage = (event) => {
        try {
          const message: TranscriptMessage = JSON.parse(event.data);
          this.onTranscriptMessage?.(message);
        } catch (error) {
          console.error('Failed to parse transcript message:', error);
        }
      };

      this.transcriptWs.onerror = (error) => {
        // Не логируем ошибки, если соединение уже закрыто
        if (this.transcriptWs?.readyState !== WebSocket.CLOSED) {
          console.error('Transcript WebSocket error:', error);
          // Не вызываем onError для каждой ошибки WebSocket
        }
      };

      this.transcriptWs.onclose = (event) => {
        console.log('Transcript WebSocket disconnected');
        this.isConnecting = false;
        if (this.shouldReconnect && event.code !== 1000) {
          this.scheduleReconnect('transcript');
        }
      };
    } catch (error) {
      console.error('Failed to create transcript WebSocket:', error);
      this.isConnecting = false;
    }
  }

  private connectLevels() {
    try {
      if (this.levelsWs?.readyState === WebSocket.OPEN) {
        return;
      }
      
      this.levelsWs = new WebSocket('ws://127.0.0.1:8000/ws/levels');
      
      this.levelsWs.onopen = () => {
        console.log('Levels WebSocket connected');
        this.reconnectAttempts = 0;
        this.isConnecting = false;
      };

      this.levelsWs.onmessage = (event) => {
        try {
          const message: LevelsMessage = JSON.parse(event.data);
          console.log('Levels WebSocket message received:', message);
          this.onLevelsMessage?.(message);
        } catch (error) {
          console.error('Failed to parse levels message:', error);
        }
      };

      this.levelsWs.onerror = (error) => {
        // Не логируем ошибки, если соединение уже закрыто
        if (this.levelsWs?.readyState !== WebSocket.CLOSED) {
          console.error('Levels WebSocket error:', error);
          // Не вызываем onError для каждой ошибки WebSocket
        }
      };

      this.levelsWs.onclose = (event) => {
        console.log('Levels WebSocket disconnected');
        this.isConnecting = false;
        if (this.shouldReconnect && event.code !== 1000) {
          this.scheduleReconnect('levels');
        }
      };
    } catch (error) {
      console.error('Failed to create levels WebSocket:', error);
      this.isConnecting = false;
    }
  }

  private scheduleReconnect(type: 'transcript' | 'levels') {
    if (!this.shouldReconnect || this.reconnectAttempts >= this.maxReconnectAttempts) {
      return;
    }
    
    this.reconnectAttempts++;
    console.log(`Scheduling ${type} WebSocket reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
    
    setTimeout(() => {
      if (this.shouldReconnect) {
        if (type === 'transcript') {
          this.connectTranscript();
        } else {
          this.connectLevels();
        }
      }
    }, this.reconnectDelay * this.reconnectAttempts);
  }

  disconnect() {
    this.shouldReconnect = false;
    this.isConnecting = false;
    
    if (this.transcriptWs) {
      this.transcriptWs.close(1000, 'Client disconnect');
      this.transcriptWs = null;
    }
    if (this.levelsWs) {
      this.levelsWs.close(1000, 'Client disconnect');
      this.levelsWs = null;
    }
  }

  isConnected(): boolean {
    return (
      (this.transcriptWs?.readyState === WebSocket.OPEN) &&
      (this.levelsWs?.readyState === WebSocket.OPEN)
    );
  }
}
