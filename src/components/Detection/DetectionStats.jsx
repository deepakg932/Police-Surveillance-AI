import React from 'react';
import { 
  User, Shield, Clock, Users, 
  Shirt, Circle, Eye, AlertCircle 
} from 'lucide-react';

const DetectionStats = ({ stats, mode }) => {
  // Map count keys to icons and colors
  const getIconForStat = (key) => {
    const iconMap = {
      'totalPersons': Users,
      'withHelmet': Shield,
      'withBlack': Circle,
      'withShirt': Shirt,
      'total_person': Users,
      'total_wear': Shield,
      'total_black': Circle,
      'total_shirt': Shirt
    };
    return iconMap[key] || Eye;
  };

  const getColorForStat = (key) => {
    const colorMap = {
      'totalPersons': 'blue',
      'withHelmet': 'yellow',
      'withBlack': 'purple',
      'withShirt': 'green',
      'total_person': 'blue',
      'total_wear': 'yellow',
      'total_black': 'purple',
      'total_shirt': 'green'
    };
    return colorMap[key] || 'gray';
  };

  const getLabelForStat = (key) => {
    const labelMap = {
      'totalPersons': 'Total Persons',
      'withHelmet': 'With Helmet',
      'withBlack': 'With Black',
      'withShirt': 'With Shirt',
      'total_person': 'Total Persons',
      'total_wear': 'With Wear',
      'total_black': 'With Black',
      'total_shirt': 'With Shirt'
    };
    return labelMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // Filter out processingTime from stats for display
  const statEntries = Object.entries(stats).filter(([key]) => key !== 'processingTime');
  
  // Create stat items dynamically from stats object
  const statItems = statEntries.map(([key, value]) => ({
    label: getLabelForStat(key),
    value: value,
    icon: getIconForStat(key),
    color: getColorForStat(key)
  }));

  // Add processing time as a separate stat item
  if (stats.processingTime) {
    statItems.push({
      label: 'Processing Time',
      value: stats.processingTime,
      icon: Clock,
      color: 'purple'
    });
  }

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
    },
    gray: {
      bg: 'bg-gray-500/10',
      text: 'text-gray-400'
    }
  };

  if (!statItems.length) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-6 text-center">
        <p className="text-gray-400">No statistics available</p>
      </div>
    );
  }

  return (
    <div>
      {mode && (
        <div className="mb-4">
          <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">
            Mode: {mode}
          </span>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {statItems.map(({ label, value, icon: Icon, color }) => (
          <div 
            key={label} 
            className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-4"
          >
            <div className="flex items-center space-x-3">
              <div className={`p-2 ${colorClasses[color]?.bg || colorClasses.gray.bg} rounded-lg`}>
                <Icon className={`h-5 w-5 ${colorClasses[color]?.text || colorClasses.gray.text}`} />
              </div>
              <div>
                <p className="text-sm text-gray-400">{label}</p>
                <p className="text-2xl font-bold text-white">{value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DetectionStats;