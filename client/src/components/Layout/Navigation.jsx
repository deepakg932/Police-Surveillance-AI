import React from 'react';
import { Upload, Video, Search, FileText } from 'lucide-react';

const Navigation = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'upload', label: 'Upload Video', icon: Upload },
    { id: 'detection', label: 'Detection Results', icon: Video },
    // { id: 'search', label: 'Search Database', icon: Search },
    // { id: 'reports', label: 'Reports', icon: FileText }
  ];

  return (
    <div className="flex space-x-1 bg-gray-800/50 p-1 rounded-xl mb-8">
      {tabs.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          onClick={() => onTabChange(id)}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === id 
              ? 'bg-blue-600 text-white shadow-lg' 
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          <Icon className="h-4 w-4" />
          <span>{label}</span>
        </button>
      ))}
    </div>
  );
};

export default Navigation;