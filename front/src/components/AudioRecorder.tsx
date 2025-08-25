import {useEffect, useState} from 'react';
import {Card, CardContent, CardHeader, CardTitle} from './ui/card';
import {Button} from './ui/button';
import {Badge} from './ui/badge';
import {Select, SelectContent, SelectItem, SelectTrigger, SelectValue} from './ui/select';
import {Slider} from './ui/slider';
import {Textarea} from './ui/textarea';
import {Separator} from './ui/separator';
import {Clock, Copy, Mic, Pause, Play, Save, Settings, Sparkles, Square} from 'lucide-react';
import {AudioLevelMeter} from './AudioLevelMeter';
import {cn} from '../lib/utils';
import audioLogo from '../assets/audio-logo.png';

type RecordingState = 'idle' | 'recording' | 'paused';

const AudioRecorder = () => {
  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [micLevel, setMicLevel] = useState(-60);
  const [blackholeLevel, setBlackholeLevel] = useState(-60);
  const [micGain, setMicGain] = useState([1.0]);
  const [blackholeGain, setBlackholeGain] = useState([1.0]);
  const [micThreshold, setMicThreshold] = useState([0.01]);
  const [blackholeThreshold, setBlackholeThreshold] = useState([0.01]);
  const [selectedMic, setSelectedMic] = useState("MacBook Pro Microphone");
  const [transcript, setTranscript] = useState("Собеседник: Привет! Как дела?\nВы: Отлично, спасибо! А у вас?\nСобеседник: Тоже хорошо. Давайте обсудим проект...\nСобеседник: У нас есть несколько интересных идей для реализации\nВы: Звучит замечательно! Расскажите подробнее\nСобеседник: Мы хотим создать современный интерфейс с использованием новых технологий");
  const [summary, setSummary] = useState("");
  const [recordingTime, setRecordingTime] = useState(0);

  // Simulate audio levels when recording
  useEffect(() => {
    if (recordingState === 'recording') {
      const interval = setInterval(() => {
        // Simulate realistic audio levels
        setMicLevel(-60 + Math.random() * 50 + Math.sin(Date.now() / 1000) * 10);
        setBlackholeLevel(-60 + Math.random() * 40 + Math.cos(Date.now() / 800) * 8);
      }, 100);
      
      return () => clearInterval(interval);
    } else {
      setMicLevel(-60);
      setBlackholeLevel(-60);
    }
  }, [recordingState]);

  // Recording timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (recordingState === 'recording') {
      interval = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    }
    
    if (recordingState === 'idle') {
      setRecordingTime(0);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [recordingState]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const mockMicrophones = [
    "MacBook Pro Microphone",
    "USB Headset",
    "Blue Yeti",
    "External Audio Interface"
  ];

  const toggleRecording = () => {
    if (recordingState === 'idle') {
      setRecordingState('recording');
    } else if (recordingState === 'recording') {
      setRecordingState('paused');
    } else {
      setRecordingState('recording');
    }
  };

  const stopRecording = () => {
    setRecordingState('idle');
  };

  const handleSummarize = () => {
    if (transcript.trim()) {
      // Simulate AI summarization
      setSummary("Краткое содержание разговора:\n\nВ ходе беседы обсуждались планы по реализации проекта с использованием современных технологий. Собеседники выразили заинтересованность в создании инновационного интерфейса и готовность к дальнейшему сотрудничеству.\n\nКлючевые моменты:\n• Обсуждение новых идей\n• Планирование технической реализации\n• Положительная реакция на предложения");
    }
  };

  const getStatusInfo = () => {
    switch (recordingState) {
      case 'recording':
        return { text: 'Запись', color: 'bg-audio-green' };
      case 'paused':
        return { text: 'Пауза', color: 'bg-audio-yellow' };
      default:
        return { text: 'Готов', color: 'bg-muted' };
    }
  };

  const status = getStatusInfo();

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <img 
                  src={audioLogo} 
                  alt="RRKOR Audio Logo" 
                  className="w-12 h-12 rounded-xl shadow-glow"
                />
                {recordingState === 'recording' && (
                  <div className="absolute -inset-1 bg-gradient-primary rounded-xl animate-glow-pulse -z-10" />
                )}
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">RRKOR™ Rec&Sum</h1>
                <p className="text-sm text-muted-foreground">Версия 0.1.1 - Professional Audio Suite</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {recordingState === 'recording' && (
              <div className="flex items-center space-x-2 px-3 py-1 bg-audio-bg rounded-lg border border-border">
                <Clock className="w-4 h-4 text-audio-green" />
                <span className="font-mono text-sm text-foreground">
                  {formatTime(recordingTime)}
                </span>
              </div>
            )}
            <Badge variant="outline" className={cn("text-white", status.color)}>
              <div className="w-2 h-2 rounded-full bg-current mr-2 animate-pulse" />
              {status.text}
            </Badge>
            <Button variant="outline" size="sm">
              <Settings className="w-4 h-4" />
            </Button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Audio Controls */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Microphone Settings */}
            <Card className="bg-gradient-card border-border shadow-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Mic className="w-5 h-5 text-primary" />
                  Настройки микрофона
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-2 block">
                    Источник
                  </label>
                  <Select value={selectedMic} onValueChange={setSelectedMic}>
                    <SelectTrigger className="bg-input border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {mockMicrophones.map((mic) => (
                        <SelectItem key={mic} value={mic}>
                          {mic}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* Mic Level Control */}
            <Card className="bg-gradient-card border-border shadow-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center justify-between">
                  <span>Микрофон</span>
                  <span className="text-xs text-muted-foreground">{micLevel.toFixed(1)} dB</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <AudioLevelMeter level={micLevel} />
                
                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-medium text-muted-foreground mb-2 block">
                      Усиление: {micGain[0].toFixed(1)}x
                    </label>
                    <Slider
                      value={micGain}
                      onValueChange={setMicGain}
                      min={0}
                      max={2}
                      step={0.1}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="text-xs font-medium text-muted-foreground mb-2 block">
                      Порог шума: {micThreshold[0].toFixed(3)}
                    </label>
                    <Slider
                      value={micThreshold}
                      onValueChange={setMicThreshold}
                      min={0}
                      max={0.1}
                      step={0.001}
                      className="w-full"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* BlackHole Level Control */}
            <Card className="bg-gradient-card border-border shadow-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center justify-between">
                  <span>BlackHole</span>
                  <span className="text-xs text-muted-foreground">{blackholeLevel.toFixed(1)} dB</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <AudioLevelMeter level={blackholeLevel} />
                
                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-medium text-muted-foreground mb-2 block">
                      Усиление: {blackholeGain[0].toFixed(1)}x
                    </label>
                    <Slider
                      value={blackholeGain}
                      onValueChange={setBlackholeGain}
                      min={0}
                      max={2}
                      step={0.1}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="text-xs font-medium text-muted-foreground mb-2 block">
                      Порог шума: {blackholeThreshold[0].toFixed(3)}
                    </label>
                    <Slider
                      value={blackholeThreshold}
                      onValueChange={setBlackholeThreshold}
                      min={0}
                      max={0.1}
                      step={0.001}
                      className="w-full"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Control Buttons */}
            <Card className="bg-gradient-card border-border shadow-card">
              <CardContent className="pt-6">
                <div className="flex flex-wrap gap-2">
                  <Button
                    onClick={toggleRecording}
                    variant={recordingState === 'recording' ? 'pause' : 'audio'}
                    className="flex-1 min-w-[100px]"
                  >
                    {recordingState === 'recording' ? (
                      <><Pause className="w-4 h-4 mr-2" />Пауза</>
                    ) : (
                      <><Play className="w-4 h-4 mr-2" />
                      {recordingState === 'paused' ? 'Продолжить' : 'Запись'}</>
                    )}
                  </Button>
                  
                  <Button
                    onClick={stopRecording}
                    variant="destructive"
                    disabled={recordingState === 'idle'}
                    className="flex-1 min-w-[100px]"
                  >
                    <Square className="w-4 h-4 mr-2" />
                    Стоп
                  </Button>
                </div>
                
                <Separator className="my-4" />
                
                <div className="flex flex-wrap gap-2">
                  <Button variant="outline" size="sm" className="flex-1">
                    <Save className="w-4 h-4 mr-2" />
                    Сохранить
                  </Button>
                  <Button variant="outline" size="sm" className="flex-1">
                    <Copy className="w-4 h-4 mr-2" />
                    Копировать
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Transcript and Summary */}
          <div className="lg:col-span-2 grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Transcript */}
            <div className="lg:col-span-2">
              <Card className="bg-gradient-card border-border shadow-card h-[600px]">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">Транскрипт</CardTitle>
                </CardHeader>
                <CardContent className="h-full pb-6">
                  <Textarea
                    value={transcript}
                    onChange={(e) => setTranscript(e.target.value)}
                    className="h-full resize-none bg-audio-bg border-border text-foreground font-mono text-sm custom-scrollbar leading-relaxed"
                    placeholder="Транскрипт появится здесь во время записи..."
                  />
                </CardContent>
              </Card>
            </div>

            {/* Summary */}
            <div className="lg:col-span-1">
              <Card className="bg-gradient-card border-border shadow-card h-[600px]">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center justify-between">
                    Саммари
                    <Button
                      size="sm"
                      onClick={handleSummarize}
                      className="bg-gradient-accent hover:bg-accent/80"
                      disabled={!transcript.trim() || recordingState !== 'idle'}
                    >
                      <Sparkles className="w-4 h-4 mr-2" />
                      Создать
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent className="h-full pb-6">
                  <Textarea
                    value={summary}
                    onChange={(e) => setSummary(e.target.value)}
                    className="h-full resize-none bg-audio-bg border-border text-foreground text-sm custom-scrollbar leading-relaxed"
                    placeholder="Нажмите 'Создать' для автоматического саммари..."
                  />
                </CardContent>
              </Card>
            </div>
          </div>
        </div>

        {/* Status Bar */}
        <div className="flex items-center justify-between text-xs text-muted-foreground bg-card border border-border rounded-lg px-4 py-2">
          <span>Готов к записи</span>
          <span>GigaAM v2 RNNT • GigaChat 2-max</span>
        </div>
      </div>
    </div>
  );
};

export default AudioRecorder;