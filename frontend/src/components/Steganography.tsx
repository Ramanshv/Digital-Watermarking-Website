import React, { useState } from 'react';
import {
  Lock,
  ArrowLeft,
  Upload,
  FileCheck,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Image as ImageIcon,
  Download, // Added Download icon
} from 'lucide-react';

// --- UI Component Replacements ---
// In a typical project, these would be imported from a UI library like shadcn/ui.
// For this single-file environment, we define basic styled components.

const Card = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={`rounded-lg border bg-card text-card-foreground shadow-sm ${className} bg-slate-800/50 border-slate-700`}
    {...props}
  />
);

const CardHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`flex flex-col space-y-1.5 p-6 ${className}`} {...props} />
);

const CardTitle = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h3
    className={`text-2xl font-semibold leading-none tracking-tight ${className} text-white`}
    {...props}
  />
);

const CardContent = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`p-6 pt-0 ${className}`} {...props} />
);

const Button = ({
  className,
  variant,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'ghost' | 'default' | 'destructive';
}) => (
  <button
    className={`inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 text-white
      ${
        variant === 'ghost'
          ? 'hover:bg-slate-700 hover:text-white text-slate-300'
          : 'bg-primary text-primary-foreground hover:bg-primary/90'
      }
      ${className}`}
    {...props}
  />
);

const Label = ({
  className,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) => (
  <label
    className={`text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 ${className} text-slate-300`}
    {...props}
  />
);

const Input = ({
  className,
  type,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) => (
  <input
    type={type}
    className={`flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50
      ${className} bg-slate-900/50 border-slate-600 text-slate-300 placeholder:text-slate-500 focus-visible:ring-slate-600`}
    {...props}
  />
);

const Alert = ({
  className,
  variant,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & {
  variant?: 'default' | 'destructive';
}) => (
  <div
    role="alert"
    className={`relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4
      ${
        variant === 'destructive'
          ? 'bg-red-900/20 border-red-700 text-red-400'
          : 'text-foreground'
      }
      ${className}`}
    {...props}
  />
);

const AlertTitle = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h5
    className={`mb-1 font-medium leading-none tracking-tight ${className}`}
    {...props}
  />
);

const AlertDescription = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`text-sm [&_p]:leading-relaxed ${className}`} {...props} />
);

// --- End of UI Component Replacements ---

interface SteganographyProps {
  onBack: () => void;
}

const Steganography = ({ onBack }: SteganographyProps) => {
  const [mode, setMode] = useState<'encode' | 'decode' | null>(null);
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [secretMessage, setSecretMessage] = useState('');
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setInputImage(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  // Function to trigger download
  const handleDownload = (base64Data: string, filename: string) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${base64Data}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleProcess = () => {
    if (!inputImage) {
      setError('Please upload an image first.');
      return;
    }

    if (mode === 'encode' && !secretMessage) {
      setError('Please enter a secret message to encode.');
      return;
    }

    setIsProcessing(true);
    setResult(null);
    setError(null);

    // This is a mock processing function.
    // In a real app, you would implement the steganography logic here.
    setTimeout(() => {
      if (mode === 'encode') {
        setResult({
          success: true,
          message: 'Message successfully hidden in the image!',
          imageB64: 'iVBORw0KGgoAAAANSUhEUgAAAAUA...', // mock output
        });
      } else {
        setResult({
          success: true,
          message: 'Hidden message extracted successfully!',
          extractedText: 'This is your secret message',
        });
      }
      setIsProcessing(false);
    }, 800);
  };

  const reset = () => {
    setMode(null);
    setInputImage(null);
    setSecretMessage('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-2xl mx-auto">
      <Button
        variant="ghost"
        className="text-slate-300 hover:text-white mb-6"
        onClick={onBack}
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back
      </Button>

      {!mode && (
        <Card className="bg-slate-800/50 border-slate-700 text-center p-8">
          <CardHeader>
            <div className="w-16 h-16 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Lock className="w-8 h-8 text-white" />
            </div>
            <CardTitle className="text-white text-3xl mb-2">
              Steganography
            </CardTitle>
            <p className="text-slate-400 mb-6">
              Choose an operation below to get started.
            </p>
            {/* Increased gap from gap-6 to gap-8 */}
            <div className="flex justify-center gap-8">
              <Button
                onClick={() => setMode('encode')}
                /* Changed button color to purple gradient */
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 px-6"
              >
                Encode
              </Button>
              <Button
                onClick={() => setMode('decode')}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 px-6"
              >
                Decode
              </Button>
            </div>
          </CardHeader>
        </Card>
      )}

      {mode && (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white text-2xl capitalize">
              {mode === 'encode' ? 'Encode Message' : 'Decode Message'}
            </CardTitle>
          </CardHeader>

          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="image" className="text-slate-300">
                {mode === 'encode' ? 'Base Image' : 'Encoded Image'}{' '}
                <span className="text-red-400">*</span>
              </Label>
              <Input
                id="image"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="bg-slate-900/50 border-slate-600 text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700"
              />
              {inputImage && (
                <div className="flex items-center gap-2 text-green-400 text-sm">
                  <FileCheck className="w-4 h-4" />
                  <span>{inputImage.name}</span>
                </div>
              )}
            </div>

            {mode === 'encode' && (
              <div className="space-y-2">
                <Label htmlFor="message" className="text-slate-300">
                  Secret Message <span className="text-red-400">*</span>
                </Label>
                <Input
                  id="message"
                  type="text"
                  value={secretMessage}
                  onChange={(e) => setSecretMessage(e.target.value)}
                  placeholder="Enter your hidden text..."
                  className="bg-slate-900/50 border-slate-600 text-slate-300"
                />
              </div>
            )}

            <Button
              onClick={handleProcess}
              disabled={isProcessing}
              className="w-full bg-gradient-to-r from-yellow-500 to-orange-600 hover:from-yellow-600 hover:to-orange-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                  Processing...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  {mode === 'encode' ? 'Encode Message' : 'Decode Message'}
                </>
              )}
            </Button>

            {error && (
              <Alert
                variant="destructive"
                className="bg-red-900/20 border-red-700"
              >
                <AlertTriangle className="h-4 w-4 text-red-400" />
                <AlertTitle className="text-red-400">Error</AlertTitle>
                <AlertDescription className="text-red-300">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {result && (
              <Alert
                className={
                  result.success
                    ? 'bg-green-900/20 border-green-700'
                    : 'bg-yellow-900/20 border-yellow-700'
                }
              >
                {result.success ? (
                  <CheckCircle className="h-4 w-4 text-green-400" />
                ) : (
                  <XCircle className="h-4 w-4 text-yellow-400" />
                )}
                <AlertTitle
                  className={
                    result.success ? 'text-green-400' : 'text-yellow-400'
                  }
                >
                  {mode === 'encode' ? 'Encoding Result' : 'Decoding Result'}
                </AlertTitle>
                <AlertDescription
                  className={
                    result.success ? 'text-green-300' : 'text-yellow-300'
                  }
                >
                  <div className="space-y-2 mt-2">
                    <p>{result.message}</p>

                    {mode === 'decode' && result.extractedText && (
                      <p>
                        <strong>Extracted Message:</strong>{' '}
                        {result.extractedText}
                      </p>
                    )}

                    {mode === 'encode' && result.imageB64 && (
                      <div className="mt-4 pt-4 border-t border-slate-700 space-y-3">
                        <h4 className="font-semibold text-slate-200 mb-2 flex items-center">
                          <ImageIcon className="w-4 h-4 mr-2" />
                          Encoded Image (Preview)
                        </h4>
                        <img
                          src={`data:image/png;base64,${result.imageB64}`}
                          alt="Encoded Output"
                          className="rounded border border-slate-600 max-w-full h-auto"
                        />
                        {/* --- Added Download Button --- */}
                        <Button
                          onClick={() =>
                            handleDownload(
                              result.imageB64,
                              'encoded-image.png'
                            )
                          }
                          className="w-full bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700"
                        >
                          <Download className="w-4 h-4 mr-2" />
                          Download Encoded Image
                        </Button>
                      </div>
                    )}
                  </div>
                </AlertDescription>
              </Alert>
            )}

            <Button
              onClick={reset}
              variant="ghost"
              className="text-slate-400 hover:text-white w-full mt-4"
            >
              Reset
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Steganography;

