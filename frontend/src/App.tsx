import { useState } from 'react';
import { MainMenu } from './components/MainMenu';
import { EmbedWatermarkFlow } from './components/EmbedWatermarkFlow';
import { DetectWatermarkFlow } from './components/DetectWatermarkFlow';
import {Steganography} from './components/Steganography';

type MainOption = 'embed' | 'detect' | 'steganography' | null;

export default function App() {
  const [selectedOption, setSelectedOption] = useState<MainOption>(null);

  const handleBack = () => {
    setSelectedOption(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {selectedOption === null && (
          <MainMenu onSelectOption={setSelectedOption} />
        )}
        
        {selectedOption === 'embed' && (
          <EmbedWatermarkFlow onBack={handleBack} />
        )}
        
        {selectedOption === 'detect' && (
          <DetectWatermarkFlow onBack={handleBack} />
        )}
        
        {selectedOption === 'steganography' && (
          <Steganography onBack={handleBack} />
        )}
      </div>
    </div>
  );
}
