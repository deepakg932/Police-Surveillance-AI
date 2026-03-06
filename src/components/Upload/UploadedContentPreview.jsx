import React from 'react';
import { Video, Image, FileText } from 'lucide-react';

const UploadedContentPreview = ({ content }) => {
  if (!content) return null;

  const getIcon = () => {
    switch(content.type) {
      case 'video': return <Video className="h-8 w-8 text-blue-400" />;
      case 'image': return <Image className="h-8 w-8 text-green-400" />;
      case 'text': return <FileText className="h-8 w-8 text-yellow-400" />;
      default: return null;
    }
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-4">
      <h3 className="text-lg font-semibold text-white mb-3">Uploaded Content</h3>
      <div className="flex items-start space-x-4">
        <div className="p-2 bg-gray-700 rounded-lg">
          {getIcon()}
        </div>
        <div className="flex-1">
          <p className="text-white">
            Type: <span className="text-blue-400 capitalize">{content.type}</span>
          </p>
          {content.file && (
            <>
              <p className="text-sm text-gray-400">
                File: {content.file.name}
              </p>
              <p className="text-xs text-gray-500">
                Size: {(content.file.size / 1024).toFixed(2)} KB
              </p>
            </>
          )}
          {content.type === 'text' && (
            <div className="mt-2">
              <p className="text-sm text-gray-400 mb-1">Content Preview:</p>
              <div className="bg-gray-700/50 p-2 rounded-lg">
                <p className="text-xs text-gray-300">
                  {content.content.substring(0, 200)}
                  {content.content.length > 200 && '...'}
                </p>
              </div>
            </div>
          )}
        </div>
        {content.type === 'image' && content.preview && (
          <img 
            src={content.preview} 
            alt="Preview" 
            className="h-20 w-20 object-cover rounded-lg border-2 border-gray-600"
          />
        )}
      </div>
    </div>
  );
};

export default UploadedContentPreview;