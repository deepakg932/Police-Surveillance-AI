import React from 'react';
import { Upload, Video, History, Search, FileText, Loader2 } from 'lucide-react';

const Navigation = ({ activeTab, onTabChange, isUploading, uploadProgress }) => {
  const tabs = [
    { id: 'upload', label: 'Upload Video', icon: Upload },
    { id: 'current', label: 'Current Results', icon: Video },
    { id: 'history', label: 'History', icon: History },
    // { id: 'search', label: 'Search Database', icon: Search },
    // { id: 'reports', label: 'Reports', icon: FileText }
  ];

  return (
    <div className="flex items-center justify-between mb-8">
      <div className="flex space-x-1 bg-gray-800/50 p-1 rounded-xl">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onTabChange(id)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
              activeTab === id 
                ? id === 'current' 
                  ? 'bg-blue-600 text-white shadow-lg'
                  : id === 'history'
                  ? 'bg-purple-600 text-white shadow-lg'
                  : 'bg-blue-600 text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            <Icon className="h-4 w-4" />
            <span>{label}</span>
          </button>
        ))}
      </div>
      
      {isUploading && (
        <div className="flex items-center space-x-2 bg-blue-500/10 px-3 py-1.5 rounded-lg border border-blue-500/30">
          <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />
          <span className="text-sm text-blue-400">Uploading... {uploadProgress}%</span>
        </div>
      )}
    </div>
  );
};

export default Navigation;