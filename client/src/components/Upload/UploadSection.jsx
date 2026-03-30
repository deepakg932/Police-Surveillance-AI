// components/Upload/UploadSection.jsx
import React from 'react';
import VideoUploadForm from './VideoUploadForm';

function UploadSection({ onSubmit }) {
  return (
    <VideoUploadForm
      onSubmit={onSubmit}
    />
  );
}

export default UploadSection;