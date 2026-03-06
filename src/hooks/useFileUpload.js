import { useState } from 'react';

export const useFileUpload = () => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [error, setError] = useState(null);

  const handleFileUpload = (e, onComplete) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file size based on type
      if (file.type.startsWith('video/') && file.size > 2 * 1024 * 1024 * 1024) {
        setError('Video file size exceeds 2GB limit');
        return;
      }
      if (file.type.startsWith('image/') && file.size > 10 * 1024 * 1024) {
        setError('Image file size exceeds 10MB limit');
        return;
      }

      setUploadedFile(file);
      setIsUploading(true);
      setError(null);
      
      // Simulate upload progress
      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        setUploadProgress(progress);
        if (progress >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          if (onComplete) onComplete(file);
        }
      }, 500);
    }
  };

  const resetUpload = () => {
    setUploadProgress(0);
    setIsUploading(false);
    setUploadedFile(null);
    setError(null);
  };

  return {
    uploadProgress,
    isUploading,
    uploadedFile,
    error,
    handleFileUpload,
    resetUpload
  };
};