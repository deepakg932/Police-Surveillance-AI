// Updated DetectionStats component
import React from 'react';
import { User, Shield, Clock, Users } from 'lucide-react';

const DetectionStats = ({ stats }) => {
  const statItems = [
    { label: 'Total Detections', value: stats.totalPersons, icon: Users, color: 'blue' },
    { label: 'With Helmet', value: stats.withHelmet, icon: Shield, color: 'yellow' },
    { label: 'Motorcycles', value: stats.totalMotorcycles || 0, icon: User, color: 'green' },
    { label: 'Processing Time', value: stats.processingTime || 'N/A', icon: Clock, color: 'purple' }
  ];

  // Map of colors to their actual classes
  const colorClasses = {
    blue: {
      bg: 'bg-blue-500/10',
      text: 'text-blue-400'
    },
    green: {
      bg: 'bg-green-500/10',
      text: 'text-green-400'
    },
    yellow: {
      bg: 'bg-yellow-500/10',
      text: 'text-yellow-400'
    },
    purple: {
      bg: 'bg-purple-500/10',
      text: 'text-purple-400'
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      {statItems.map(({ label, value, icon: Icon, color }) => (
        <div key={label} className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-4">
          <div className="flex items-center space-x-3">
            <div className={`p-2 ${colorClasses[color].bg} rounded-lg`}>
              <Icon className={`h-5 w-5 ${colorClasses[color].text}`} />
            </div>
            <div>
              <p className="text-sm text-gray-400">{label}</p>
              <p className="text-2xl font-bold text-white">{value}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default DetectionStats;