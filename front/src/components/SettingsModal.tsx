import { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Settings, Save, ArrowLeft } from 'lucide-react';
import { cn } from '../lib/utils';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (settings: GigaChatSettings) => void;
  currentSettings: GigaChatSettings;
}

export interface GigaChatSettings {
  credentials: string;
  scope: string;
  model: string;
}

const GIGACHAT_MODELS = [
  'GigaChat-2-Max',
  'GigaChat-2-Pro', 
  'GigaChat-2-Lite'
];

export const SettingsModal = ({ isOpen, onClose, onSave, currentSettings }: SettingsModalProps) => {
  const [settings, setSettings] = useState<GigaChatSettings>(currentSettings);
  const [isSaving, setIsSaving] = useState(false);

  // Обновляем локальное состояние при изменении пропсов
  useEffect(() => {
    setSettings(currentSettings);
  }, [currentSettings]);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await onSave(settings);
      onClose();
    } catch (error) {
      console.error('Ошибка сохранения настроек:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    // Восстанавливаем исходные значения
    setSettings(currentSettings);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[500px] bg-gradient-card border-border shadow-card dialog-content">
        <DialogHeader className="flex-shrink-0">
          <DialogTitle className="flex items-center gap-2 text-xl font-bold">
            <Settings className="w-6 h-6 text-primary" />
            Настройки GigaChat
          </DialogTitle>
        </DialogHeader>
        
        <div className="dialog-body space-y-6 py-4">
          {/* GigaChat Credentials */}
          <div className="space-y-2">
            <Label htmlFor="credentials" className="text-sm font-medium text-foreground">
              GIGACHAT_CREDENTIALS
            </Label>
            <Input
              id="credentials"
              type="password"
              value={settings.credentials}
              onChange={(e) => setSettings(prev => ({ ...prev, credentials: e.target.value }))}
              placeholder="Введите credentials"
              className="bg-input border-border focus:border-primary"
            />
          </div>

          {/* GigaChat Scope */}
          <div className="space-y-2">
            <Label htmlFor="scope" className="text-sm font-medium text-foreground">
              GIGACHAT_SCOPE
            </Label>
            <Input
              id="scope"
              type="text"
              value={settings.scope}
              onChange={(e) => setSettings(prev => ({ ...prev, scope: e.target.value }))}
              placeholder="Введите scope"
              className="bg-input border-border focus:border-primary"
            />
          </div>

          {/* GigaChat Model */}
          <div className="space-y-2">
            <Label htmlFor="model" className="text-sm font-medium text-foreground">
              GIGACHAT_MODEL
            </Label>
            <Select
              value={settings.model}
              onValueChange={(value) => setSettings(prev => ({ ...prev, model: value }))}
            >
              <SelectTrigger className="bg-input border-border focus:border-primary">
                <SelectValue placeholder="Выберите модель" />
              </SelectTrigger>
              <SelectContent>
                {GIGACHAT_MODELS.map((model) => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="dialog-footer flex justify-end space-x-3">
          <Button
            variant="outline"
            onClick={handleCancel}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Назад
          </Button>
          <Button
            onClick={handleSave}
            disabled={isSaving}
            className={cn(
              "flex items-center gap-2 bg-gradient-primary text-white hover:bg-gradient-primary/90",
              isSaving && "opacity-50 cursor-not-allowed"
            )}
          >
            <Save className="w-4 h-4" />
            {isSaving ? 'Сохранение...' : 'Сохранить'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
