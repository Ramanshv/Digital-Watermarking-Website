import { useState } from 'react';
import { Upload, Image as ImageIcon, FileCheck, AlertTriangle, Download } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';

interface EmbedFormProps {
  method: 'dct' | 'dwt' | 'hybrid';
}

export function EmbedForm({ method }: EmbedFormProps) {
  const [coverImage, setCoverImage] = useState<File | null>(null);
  const [watermarkLogo, setWatermarkLogo] = useState<File | null>(null);
  // 'alpha' state has been removed
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<string | null>(null); // Will store the blob URL
  const [error, setError] = useState<string | null>(null); // Added state for errors

  const methodTitles = {
    dct: 'DCT',
    dwt: 'DWT',
    hybrid: 'Hybrid',
  };

  const handleCoverImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setCoverImage(e.target.files[0]);
      setResult(null); // Clear previous result
      setError(null); // Clear previous error
    }
  };

  const handleWatermarkChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setWatermarkLogo(e.target.files[0]);
      setResult(null); // Clear previous result
      setError(null); // Clear previous error
    }
  };

  const handleEmbed = async () => {
    if (!coverImage || !watermarkLogo) {
      return;
    }

    setIsProcessing(true);
    setResult(null);
    setError(null);

    const apiEndpoints = {
      dct: 'http://localhost:5001/embed_dct_tiled', // You will need to create this API endpoint
      dwt: 'http://localhost:5001/embed_dwt_color', // You will need to create this API endpoint
      hybrid: 'http://localhost:5001/embed_dwt_dct', // This is the API you provided
    };

    const endpoint = apiEndpoints[method];

    // Create FormData (alpha is removed)
    const formData = new FormData();
    formData.append('host_image', coverImage);
    formData.append('watermark_image', watermarkLogo);

    // 'alpha' is no longer appended to formData

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
        // No 'Content-Type' header needed; browser sets it for FormData
      });

      // Handle API error response
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Embedding failed. Please try again.');
      }

      // Handle successful image file response
      const imageBlob = await response.blob();
      const imageUrl = URL.createObjectURL(imageBlob);
      setResult(imageUrl); // Set the result to the downloadable blob URL

    } catch (err: any) {
      setError(err.message || 'An unexpected network error occurred.');
    } finally {
      setIsProcessing(false);
    }
  };

  const canProcess = coverImage && watermarkLogo;

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl mb-2 text-white">
          Embed Watermark - {methodTitles[method]}
        </h2>
        <p className="text-slate-400">
          Upload your cover image and watermark logo to proceed
        </p>
      </div>

      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Files
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Cover Image Upload */}
          <div className="space-y-2">
            <Label htmlFor="cover-image" className="text-slate-300">
              Cover Image <span className="text-red-400">*</span>
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
          </div>

          {/* Watermark Logo Upload */}
          <div className="space-y-2">
            <Label htmlFor="watermark-logo" className="text-slate-300">
              Watermark Logo <span className="text-red-400">*</span>
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

          {/* --- Alpha (Strength) Input has been REMOVED --- */}
          
          {/* Process Button */}
          <Button
            onClick={handleEmbed}
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
                <ImageIcon className="w-4 h-4 mr-2" />
                Embed Watermark
              </>
            )}
          </Button>

          {/* Error Alert */}
          {error && (
            <Alert variant="destructive" className="bg-red-900/20 border-red-700">
              <AlertTriangle className="h-4 w-4 text-red-400" />
              <AlertTitle className="text-red-400">Error</AlertTitle>
              <AlertDescription className="text-red-300">
                {error}
              </AlertDescription>
            </Alert>
          )}

          {/* Result Alert (now shows download) */}
          {result && (
            <Alert className="bg-green-900/20 border-green-700">
              <FileCheck className="h-4 w-4 text-green-400" />
              <AlertTitle className="text-green-400">Success!</AlertTitle>
              <AlertDescription className="text-green-300 flex justify-between items-center">
                Watermark embedded successfully.
                <Button asChild variant="outline" size="sm" className="bg-green-600 text-white hover:bg-green-700 hover:text-white">
                  <a href={result} download="watermarked_output.png">
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </a>
                </Button>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
}