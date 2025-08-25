import {cn} from '../lib/utils';

interface AudioLevelMeterProps {
  level: number; // dB level from -60 to 0
  className?: string;
}

export const AudioLevelMeter = ({ level, className }: AudioLevelMeterProps) => {
  // Convert dB level (-60 to 0) to percentage (0 to 100)
  // Используем нелинейную шкалу для лучшей чувствительности к низким уровням
  const normalizedLevel = Math.max(-60, Math.min(0, level));
  const percentage = Math.max(0, Math.min(100, ((normalizedLevel + 60) / 60) * 100));
  
  // Более чувствительная цветовая схема для PPM
  const getColor = () => {
    if (percentage <= 30) return 'bg-audio-green';      // Зеленый до -42 dB
    if (percentage <= 60) return 'bg-audio-yellow';    // Желтый до -24 dB  
    if (percentage <= 85) return 'bg-audio-orange';    // Оранжевый до -9 dB
    return 'bg-audio-red';                             // Красный выше -9 dB
  };

  const getGradientColor = () => {
    if (percentage <= 30) return 'from-audio-green/20 to-audio-green';
    if (percentage <= 60) return 'from-audio-yellow/20 to-audio-yellow';
    if (percentage <= 85) return 'from-orange-400/20 to-orange-500';
    return 'from-audio-red/20 to-audio-red';
  };

  return (
    <div className={cn("relative", className)}>
      {/* Background */}
      <div className="w-full h-6 bg-audio-bg rounded-lg border border-border overflow-hidden">
        
        {/* Level Bar with Gradient */}
        <div 
          className={cn(
            "h-full bg-gradient-to-r transition-all duration-150 ease-out",
            getGradientColor()
          )}
          style={{ width: `${percentage}%` }}
        />
        
        {/* Peak indicators */}
        <div className="absolute inset-0 flex items-center">
          {/* Segment markers */}
          {[20, 40, 60, 80].map((mark) => (
            <div
              key={mark}
              className="absolute w-px h-4 bg-border/50"
              style={{ left: `${mark}%` }}
            />
          ))}
        </div>
      </div>
      
      {/* dB Scale */}
      <div className="flex justify-between text-xs text-muted-foreground mt-1">
        <span>-60</span>
        <span>-42</span>
        <span>-24</span>
        <span>-9</span>
        <span>0</span>
      </div>
      
      {/* Current Level Indicator */}
      <div className="flex justify-end mt-1">
        <span className={cn("text-xs font-mono", getColor().replace('bg-', 'text-'))}>
          {level.toFixed(1)} dB
        </span>
      </div>
    </div>
  );
};