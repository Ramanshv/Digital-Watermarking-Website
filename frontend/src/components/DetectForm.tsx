import { useState } from 'react';
import {
  Search,
  FileCheck,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Image as ImageIcon,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';

interface DetectFormProps {
  method: 'dct' | 'dwt' | 'hybrid';
}

// Define a type for the result object
interface DetectionResult {
  correlation: number;
  detected: boolean;
  threshold: number;
  imageB64?: string; // Extracted image (Base64)
}

export function DetectForm({ method }: DetectFormProps) {
  const [watermarkedImage, setWatermarkedImage] = useState<File | null>(null);
  const [watermarkLogo, setWatermarkLogo] = useState<File | null>(null);
  const [coverImage, setCoverImage] = useState<File | null>(null);

  // --- State for API parameters ---
  const [alpha, setAlpha] = useState(0.1);

  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const methodTitles = {
    dct: 'DCT',
    dwt: 'DWT',
    hybrid: 'Hybrid (DWT-DCT)',
  };

  // Check which parameters are needed based on method
  const needsCoverImage = method === 'dwt' || method === 'hybrid';
  const needsAlpha = method === 'dwt' || method === 'hybrid';

  // --- Handlers to clear results on new file selection ---
  const handleWatermarkedImageChange = (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    if (e.target.files && e.target.files[0]) {
      setWatermarkedImage(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleWatermarkChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setWatermarkLogo(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleCoverImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setCoverImage(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  // --- THIS IS THE UPDATED MOCK FUNCTION ---
  const handleDetect = () => {
    const requiredFilesPresent = needsCoverImage
      ? watermarkedImage && watermarkLogo && coverImage
      : watermarkedImage && watermarkLogo;

    if (!requiredFilesPresent) {
      setError('Please upload all required files.');
      return;
    }

    setIsProcessing(true);
    setResult(null);
    setError(null);

    // --- NO API CALL ---
    // We just simulate a delay and return the hardcoded result

    let mockResult: DetectionResult;

    switch (method) {
      case 'hybrid':
        // "Watermark detected! Correlation = 0.290"
        mockResult = {
          correlation: 0.29,
          detected: true,
          threshold: 0.25, // Default threshold
        };
        break;

      case 'dwt':
        // "Correlation (DWT): 0.7666 Watermark detected!"
        mockResult = {
          correlation: 0.7666,
          detected: true,
          threshold: 0.5, // Default threshold
        };
        break;

      case 'dct':
      default:
        // "Average correlation: 0.8007,Watermark detected!"
        // Note: The UI uses 'correlation' from the result object.
        mockResult = {
          correlation: 0.8007,
          detected: true,
          threshold: 0.05, // Default threshold
        };
        break;
    }

    // Simulate a network delay for better UX
    setTimeout(() => {
      setResult(mockResult);
      setIsProcessing(false);
    }, 500); // 500ms delay
  };

  const canProcess = needsCoverImage
    ? watermarkedImage && watermarkLogo && coverImage
    : watermarkedImage && watermarkLogo;

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl mb-2 text-white">
          Detect Watermark - {methodTitles[method]}
        </h2>
        <p className="text-slate-400">
          Upload the required files to detect and extract the watermark
        </p>
      </div>

      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Search className="w-5 h-5" />
            Upload Files & Parameters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Watermarked Image Upload */}
          <div className="space-y-2">
            <Label htmlFor="watermarked-image" className="text-slate-300">
              Watermarked Image <span className="text-red-400">*</span>
            </Label>
            <Input
              id="watermarked-image"
              type="file"
              accept="image/*"
              onChange={handleWatermarkedImageChange}
              className="bg-slate-900/50 border-slate-600 text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700"
              required
            />
            {watermarkedImage && (
              <div className="flex items-center gap-2 text-green-400 text-sm">
                <FileCheck className="w-4 h-4" />
                <span>{watermarkedImage.name}</span>
              </div>
            )}
          </div>

          {/* Watermark Logo Upload */}
          <div className="space-y-2">
            <Label htmlFor="watermark-logo" className="text-slate-300">
              Original Watermark Logo <span className="text-red-400">*</span>
            </Label>
            <Input
              id="watermark-logo"
              type="file"
              accept="image/*"
              onChange={handleWatermarkChange}
              className="bg-slate-900/50 border-slate-600 text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700"
              required
            />
            {watermarkLogo && (
              <div className="flex items-center gap-2 text-green-400 text-sm">
                <FileCheck className="w-4 h-4" />
                <span>{watermarkLogo.name}</span>
              </div>
            )}
          </div>

          {/* Cover Image Upload - Only for DWT and Hybrid */}
          {needsCoverImage && (
            <div className="space-y-2">
              <Label htmlFor="cover-image" className="text-slate-300">
                Original Cover Image <span className="text-red-400">*</span>
              </Label>
              <Input
                id="cover-image"
                type="file"
                accept="image/*"
                onChange={handleCoverImageChange}
                className="bg-slate-900/50 border-slate-600 text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700"
                required
              />
              {coverImage && (
                <div className="flex items-center gap-2 text-green-400 text-sm">
                  <FileCheck className="w-4 h-4" />
                  <span>{coverImage.name}</span>
                </div>
              )}
              <p className="text-xs text-slate-500">
                Required for {methodTitles[method]} detection method.
              </p>
            </div>
          )}

          {/* --- UPDATED: Parameter Inputs --- */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {needsAlpha && (
              <div className="space-y-2">
                <Label htmlFor="alpha" className="text-slate-300">
                  Alpha (Strength)
                </Label>
                <Input
                  id="alpha"
                  type="number"
                  value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value))}
                  step="0.01"
                  min="0"
                  max="1"
                  className="bg-slate-900/50 border-slate-600 text-slate-300"
                />
              </div>
            )}
          </div>

          {/* Process Button */}
          <Button
            onClick={handleDetect}
            disabled={!canProcess || isProcessing}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                Processing...
              </>
            ) : (
              <>
                <Search className="w-4 h-4 mr-2" />
                Detect Watermark
              </>
            )}
          </Button>

          {/* --- Error Alert --- */}
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

          {/* --- Result Alert --- */}
          {result && (
            <Alert
              className={
                result.detected
                  ? 'bg-green-900/20 border-green-700'
                  : 'bg-yellow-900/20 border-yellow-700'
              }
            >
              {result.detected ? (
                <CheckCircle className="h-4 w-4 text-green-400" />
              ) : (
                <XCircle className="h-4 w-4 text-yellow-400" />
              )}

              <AlertTitle
                className={
                  result.detected ? 'text-green-400' : 'text-yellow-400'
                }
              >
                Detection Result: {result.detected ? 'Found!' : 'Not Found'}
              </AlertTitle>

              <AlertDescription
                className={
                  result.detected ? 'text-green-300' : 'text-yellow-300'
                }
              >
                <div className="space-y-2 mt-2">
                  <p>
                    <strong>
                      {/* Show 'Average Correlation' for DCT, 'Correlation' for others */}
                      {method === 'dct'
                        ? 'Average Correlation:'
                        : 'Correlation:'}
                    </strong>{' '}
                    {result.correlation.toFixed(4)}
                  </p>
                  <p>
                    <strong>Threshold Used:</strong> {result.threshold}
                  </p>
                  <p className="text-xs text-slate-400">
                    (Correlation &gt; Threshold = Detected)
                  </p>

                  {/* Show extracted image if it exists */}
                  {result.imageB64 && (
                    <div className="mt-4 pt-4 border-t border-slate-700">
                      <h4 className="font-semibold text-slate-200 mb-2 flex items-center">
                        <ImageIcon className="w-4 h-4 mr-2" />
                        Extracted Watermark
                      </h4>
                      <img
                        src={`data:image/png;base64,${result.imageB64}`}
                        alt="Extracted Watermark"
                        className="rounded border border-slate-600 max-w-full h-auto"
                      />
                    </div>
                  )}
                </div>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
