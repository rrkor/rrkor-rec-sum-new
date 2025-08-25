import {cn} from '../lib/utils';

interface AudioLevelMeterProps {
  level: number; // dB level from -60 to 0
  className?: string;
}

export const AudioLevelMeter = ({ level, className }: AudioLevelMeterProps) => {
  // Convert dB level (-60 to 0) to percentage (0 to 100)
  const percentage = Math.max(0, Math.min(100, ((level + 60) / 60) * 100));
  
  // Determine color based on level
  const getColor = () => {
    if (percentage <= 70) return 'bg-audio-green';
    if (percentage <= 90) return 'bg-audio-yellow';
    return 'bg-audio-red';
  };

  const getGradientColor = () => {
    if (percentage <= 70) return 'from-audio-green/20 to-audio-green';
    if (percentage <= 90) return 'from-audio-yellow/20 to-audio-yellow';
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
        <span>-30</span>
        <span>-12</span>
        <span>-6</span>
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