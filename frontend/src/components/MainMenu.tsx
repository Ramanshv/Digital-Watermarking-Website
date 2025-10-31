import { Shield, Search, Lock } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';

type MainOption = 'embed' | 'detect' | 'steganography';

interface MainMenuProps {
  onSelectOption: (option: MainOption) => void;
}

export function MainMenu({ onSelectOption }: MainMenuProps) {
  const options = [
    {
      id: 'embed' as const,
      icon: Shield,
      title: 'Embed Watermark',
      description: 'Add a watermark to your image using DCT, DWT, or Hybrid methods',
      available: true,
    },
    {
      id: 'detect' as const,
      icon: Search,
      title: 'Detect Watermark',
      description: 'Extract and verify watermarks from images',
      available: true,
    },
    {
      id: 'steganography' as const,
      icon: Lock,
      title: 'Steganography',
      description: 'Hide secret messages within images',
      available: true,
    },
  ];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-5xl mb-4 bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
          CrypticCanvas
        </h1>
        <p className="text-slate-300 text-lg">
          Advanced Image Watermarking & Steganography Platform
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {options.map((option) => {
          const Icon = option.icon;
          return (
            <Card
              key={option.id}
              className="bg-slate-800/50 border-slate-700 hover:border-purple-500 transition-all duration-300 hover:scale-105 cursor-pointer"
              onClick={() => option.available && onSelectOption(option.id)}
            >
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center mb-4">
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <CardTitle className="text-white flex items-center gap-2">
                  {option.title}
                  {!option.available && (
                    <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded">
                      Coming Soon
                    </span>
                  )}
                </CardTitle>
                <CardDescription className="text-slate-400">
                  {option.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                  disabled={!option.available}
                >
                  {option.available ? 'Get Started' : 'Coming Soon'}
                </Button>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
