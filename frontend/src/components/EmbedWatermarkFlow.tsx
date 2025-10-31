import { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Button } from './ui/button';
import { MethodSelector } from './MethodSelector';
import { EmbedForm } from './EmbedForm';

type EmbedMethod = 'dct' | 'dwt' | 'hybrid' | null;

interface EmbedWatermarkFlowProps {
  onBack: () => void;
}

export function EmbedWatermarkFlow({ onBack }: EmbedWatermarkFlowProps) {
  const [selectedMethod, setSelectedMethod] = useState<EmbedMethod>(null);

  const handleBackToMethods = () => {
    setSelectedMethod(null);
  };

  const methods = [
    {
      id: 'dct' as const,
      title: 'Embed DCT',
      description: 'Discrete Cosine Transform watermarking',
    },
    {
      id: 'dwt' as const,
      title: 'Embed DWT',
      description: 'Discrete Wavelet Transform watermarking',
    },
    {
      id: 'hybrid' as const,
      title: 'Embed Hybrid',
      description: 'Combined DCT-DWT watermarking',
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
          title="Embed Watermark"
          subtitle="Choose a watermarking method to embed your watermark"
          methods={methods}
          onSelectMethod={setSelectedMethod}
        />
      ) : (
        <EmbedForm method={selectedMethod} />
      )}
    </div>
  );
}
