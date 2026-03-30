// components/Upload/VideoUploadForm.jsx
import React, { useState, useEffect } from "react";
import { Upload, Video, Image, FileText, X, Send, Plus, Loader2 } from "lucide-react";
import { useUpload } from "../contexts/UploadContext";

const VideoUploadForm = ({
  onSubmit,
}) => {
  const { 
    uploadProgress, 
    isUploading, 
    uploadData,
    startUpload, 
    updateProgress, 
    completeUpload, 
    failUpload,
    updateUploadData,
    clearUploadData
  } = useUpload();
  
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Initialize from context or use defaults
  const [mainVideo, setMainVideo] = useState(uploadData?.video || null);
  const [additionalImage, setAdditionalImage] = useState(uploadData?.image || null);
  const [additionalText, setAdditionalText] = useState(uploadData?.text || "");
  const [imagePreviewUrl, setImagePreviewUrl] = useState(uploadData?.imagePreviewUrl || null);
  const [activeTab, setActiveTab] = useState(uploadData?.activeTab || "video");

  // Update context when local state changes
  useEffect(() => {
    updateUploadData({
      video: mainVideo,
      image: additionalImage,
      text: additionalText,
      imagePreviewUrl,
      activeTab
    });
  }, [mainVideo, additionalImage, additionalText, imagePreviewUrl, activeTab, updateUploadData]);

  // Clean up preview URL on unmount
  useEffect(() => {
    return () => {
      if (imagePreviewUrl && imagePreviewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
    };
  }, [imagePreviewUrl]);

  const handleVideoSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("video/")) {
      setMainVideo(file);
    }
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image/")) {
      setAdditionalImage(file);
      if (imagePreviewUrl && imagePreviewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
      const url = URL.createObjectURL(file);
      setImagePreviewUrl(url);
    }
  };

  const handleRemoveVideo = () => {
    setMainVideo(null);
    const videoInput = document.getElementById("video-upload");
    if (videoInput) videoInput.value = "";
  };

  const handleRemoveImage = () => {
    if (imagePreviewUrl && imagePreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setAdditionalImage(null);
    setImagePreviewUrl(null);
    const imageInput = document.getElementById("image-upload");
    if (imageInput) imageInput.value = "";
  };

  const handleRemoveText = () => {
    setAdditionalText("");
  };

  const handleTextChange = (e) => {
    const newValue = e.target.value;
    setAdditionalText(newValue);
  };

  // Simulate upload progress
  const simulateUpload = async () => {
    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 200));
      updateProgress(i);
    }
  };

  const handleMainSubmit = async () => {
    if (!mainVideo) {
      alert("Please select a video first");
      return;
    }

    setIsSubmitting(true);
    startUpload();

    try {
      const formData = {
        video: mainVideo,
        ...(additionalImage && { image: additionalImage }),
        ...(additionalText && { text: additionalText }),
      };

      // Simulate upload progress
      await simulateUpload();

      if (onSubmit) {
        await onSubmit(formData);
      }
      
      completeUpload(mainVideo);
    } catch (error) {
      console.error("Submission error:", error);
      failUpload(error.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleResetForm = () => {
    if (imagePreviewUrl && imagePreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    
    setMainVideo(null);
    setAdditionalImage(null);
    setAdditionalText("");
    setImagePreviewUrl(null);
    setActiveTab("video");
    setIsSubmitting(false);
    
    clearUploadData();

    const videoInput = document.getElementById("video-upload");
    const imageInput = document.getElementById("image-upload");
    if (videoInput) videoInput.value = "";
    if (imageInput) imageInput.value = "";
  };

  const isSubmitDisabled = () => {
    return !mainVideo || isUploading || isSubmitting;
  };

  const renderVideoSection = () => (
    <div className="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center hover:border-blue-500 transition-colors">
      <input
        type="file"
        accept="video/*"
        onChange={handleVideoSelect}
        className="hidden"
        id="video-upload"
      />
      <label htmlFor="video-upload" className="cursor-pointer">
        <Video className="h-16 w-16 text-gray-500 mx-auto mb-4" />
        <p className="text-gray-300 mb-2">Click to upload main video</p>
        <p className="text-sm text-gray-500">MP4, AVI, MOV (Max 2GB)</p>
      </label>
    </div>
  );

  const renderSelectedVideo = () => {
    if (!mainVideo) return null;

    return (
      <div className="mt-4 p-4 bg-gray-700/30 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">Main Video:</span>
          <button
            onClick={handleRemoveVideo}
            className="p-1 hover:bg-gray-600 rounded-full transition-colors"
            type="button"
            disabled={isSubmitting || isUploading}
          >
            <X className="h-4 w-4 text-gray-400" />
          </button>
        </div>

        <div className="flex items-center space-x-3">
          <Video className="h-12 w-12 text-blue-400" />
          <div>
            <p className="text-white text-sm">{mainVideo.name}</p>
            <p className="text-xs text-gray-500">
              {(mainVideo.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>
        </div>
      </div>
    );
  };

  const renderAdditionalContent = () => (
    <div className="mt-6 space-y-4">
      <h3 className="text-lg font-semibold text-white">
        Additional Content (Optional)
      </h3>

      <div className="flex space-x-2 mb-4">
        <button
          onClick={() => setActiveTab("image")}
          disabled={isSubmitting || isUploading}
          className={`flex-1 flex items-center justify-center space-x-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === "image"
              ? "bg-green-600 text-white"
              : "bg-gray-700 text-gray-300 hover:bg-gray-600"
          } ${isSubmitting || isUploading ? "opacity-50 cursor-not-allowed" : ""}`}
          type="button"
        >
          <Image className="h-4 w-4" />
          <span>Add Image</span>
        </button>

        <button
          onClick={() => setActiveTab("text")}
          disabled={isSubmitting || isUploading}
          className={`flex-1 flex items-center justify-center space-x-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === "text"
              ? "bg-green-600 text-white"
              : "bg-gray-700 text-gray-300 hover:bg-gray-600"
          } ${isSubmitting || isUploading ? "opacity-50 cursor-not-allowed" : ""}`}
          type="button"
        >
          <FileText className="h-4 w-4" />
          <span>Add Text</span>
        </button>
      </div>

      {activeTab === "image" && (
        <div>
          {!additionalImage ? (
            <div className="border-2 border-dashed border-gray-600 rounded-xl p-6 text-center hover:border-green-500 transition-colors">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
                id="image-upload"
                disabled={isSubmitting || isUploading}
              />
              <label 
                htmlFor="image-upload" 
                className={`cursor-pointer ${isSubmitting || isUploading ? "opacity-50" : ""}`}
              >
                <Image className="h-12 w-12 text-gray-500 mx-auto mb-2" />
                <p className="text-gray-300 text-sm">Click to upload image</p>
                <p className="text-xs text-gray-500 mt-1">
                  JPG, PNG, GIF (Max 10MB)
                </p>
              </label>
            </div>
          ) : (
            <div className="p-4 bg-gray-700/30 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Additional Image:</span>
                <button
                  onClick={handleRemoveImage}
                  className="p-1 hover:bg-gray-600 rounded-full transition-colors"
                  type="button"
                  disabled={isSubmitting || isUploading}
                >
                  <X className="h-4 w-4 text-gray-400" />
                </button>
              </div>

              <div className="flex items-center space-x-3">
                {imagePreviewUrl && (
                  <img
                    src={imagePreviewUrl}
                    alt="Preview"
                    className="h-16 w-16 object-cover rounded-lg"
                  />
                )}
                <div>
                  <p className="text-white text-sm">{additionalImage.name}</p>
                  <p className="text-xs text-gray-500">
                    {(additionalImage.size / 1024).toFixed(2)} KB
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "text" && (
        <div>
          <div className="border-2 border-gray-600 rounded-xl p-4">
            <textarea
              value={additionalText}
              onChange={handleTextChange}
              placeholder="Enter additional text here... (You can type multiple characters)"
              className="w-full h-32 px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-green-500 resize-none"
              autoFocus
              disabled={isSubmitting || isUploading}
            />
            <div className="flex justify-between items-center mt-2">
              <span className="text-xs text-gray-400">
                {additionalText.length} characters
              </span>
              {additionalText.length > 0 && (
                <button
                  onClick={handleRemoveText}
                  className="text-xs text-red-400 hover:text-red-300"
                  disabled={isSubmitting || isUploading}
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          {additionalText && (
            <div className="mt-4 p-4 bg-gray-700/30 rounded-lg">
              <div className="flex items-start space-x-3">
                <FileText className="h-8 w-8 text-green-400 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-white text-sm mb-2">Text Preview:</p>
                  <div className="bg-gray-800 p-3 rounded-lg">
                    <p className="text-sm text-gray-300 whitespace-pre-wrap break-words">
                      {additionalText}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {(additionalImage || additionalText) && (
        <div className="flex items-center space-x-2 text-sm text-green-400 bg-green-500/10 p-2 rounded-lg">
          <Plus className="h-4 w-4" />
          <span>
            {additionalImage && "🖼️ Image "}
            {additionalImage && additionalText && "+ "}
            {additionalText && "📝 Text "}
            will be sent with video
          </span>
        </div>
      )}
    </div>
  );

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-8">
          <div className="inline-block p-4 bg-blue-500/10 rounded-full mb-4">
            <Upload className="h-12 w-12 text-blue-400" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Upload Media</h2>
          <p className="text-gray-400">
            Upload video with optional image or text
          </p>
        </div>

        {!mainVideo ? renderVideoSection() : renderSelectedVideo()}

        {mainVideo && renderAdditionalContent()}

        {mainVideo && (
          <div className="mt-8">
            <button
              onClick={handleMainSubmit}
              disabled={isSubmitDisabled()}
              className="w-full py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shadow-lg"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Submitting...</span>
                </>
              ) : isUploading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Uploading... {uploadProgress}%</span>
                </>
              ) : (
                <>
                  <Send className="h-5 w-5" />
                  <span>
                    Upload Video {additionalImage || additionalText ? "with Attachments" : ""}
                  </span>
                </>
              )}
            </button>

            <button
              onClick={handleResetForm}
              className="w-full mt-2 py-2 text-gray-400 hover:text-white transition-colors text-sm"
              type="button"
              disabled={isSubmitting || isUploading}
            >
              Reset Form
            </button>
          </div>
        )}

        {isUploading && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-400 mb-2">
              <span>Uploading...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">
              Processing your video{additionalImage ? " and image" : ""}
              {additionalText ? " and text" : ""}...
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoUploadForm;