import { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Button } from './ui/button';
import { MethodSelector } from './MethodSelector';
import { DetectForm } from './DetectForm';

type DetectMethod = 'dct' | 'dwt' | 'hybrid' | null;

interface DetectWatermarkFlowProps {
  onBack: () => void;
}

export function DetectWatermarkFlow({ onBack }: DetectWatermarkFlowProps) {
  const [selectedMethod, setSelectedMethod] = useState<DetectMethod>(null);

  const handleBackToMethods = () => {
    setSelectedMethod(null);
  };

  const methods = [
    {
      id: 'dct' as const,
      title: 'Detect DCT',
      description: 'Extract DCT-based watermark',
    },
    {
      id: 'dwt' as const,
      title: 'Detect DWT',
      description: 'Extract DWT-based watermark',
    },
    {
      id: 'hybrid' as const,
      title: 'Detect Hybrid',
      description: 'Extract Hybrid watermark',
    },
  ];

  return (
    <div className="max-w-4xl mx-auto">
      <Button
        variant="ghost"
        className="text-slate-300 hover:text-white mb-6"
        onClick={selectedMethod ? handleBackToMethods : onBack}
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back
      </Button>

      {selectedMethod === null ? (
        <MethodSelector
          title="Detect Watermark"
          subtitle="Choose a detection method to extract the watermark"
          methods={methods}
          onSelectMethod={setSelectedMethod}
        />
      ) : (
        <DetectForm method={selectedMethod} />
      )}
    </div>
  );
}
