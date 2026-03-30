// components/contexts/UploadContext.jsx
import React, { createContext, useContext, useState, useCallback, useRef } from 'react';

const UploadContext = createContext(null);

export function UploadProvider({ children }) {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadData, setUploadData] = useState({
    video: null,
    image: null,
    text: '',
    imagePreviewUrl: null,
    activeTab: 'video'
  });

  const startUpload = useCallback(() => {
    setIsUploading(true);
    setUploadProgress(0);
    setUploadError(null);
  }, []);

  const updateProgress = useCallback((progress) => {
    setUploadProgress(progress);
  }, []);

  const completeUpload = useCallback((fileData) => {
    setIsUploading(false);
    setUploadProgress(100);
    setUploadedFile(fileData);
  }, []);

  const failUpload = useCallback((error) => {
    setIsUploading(false);
    setUploadError(error);
    setUploadProgress(0);
  }, []);

  const updateUploadData = useCallback((data) => {
    setUploadData(prev => ({ ...prev, ...data }));
  }, []);

  const resetUpload = useCallback(() => {
    setIsUploading(false);
    setUploadProgress(0);
    setUploadError(null);
    setUploadedFile(null);
    setUploadData({
      video: null,
      image: null,
      text: '',
      imagePreviewUrl: null,
      activeTab: 'video'
    });
  }, []);

  const clearUploadData = useCallback(() => {
    setUploadData({
      video: null,
      image: null,
      text: '',
      imagePreviewUrl: null,
      activeTab: 'video'
    });
  }, []);

  const value = {
    uploadProgress,
    isUploading,
    uploadError,
    uploadedFile,
    uploadData,
    startUpload,
    updateProgress,
    completeUpload,
    failUpload,
    resetUpload,
    updateUploadData,
    clearUploadData
  };

  return (
    <UploadContext.Provider value={value}>
      {children}
    </UploadContext.Provider>
  );
}

export function useUpload() {
  const context = useContext(UploadContext);
  if (!context) {
    throw new Error('useUpload must be used within an UploadProvider');
  }
  return context;
}