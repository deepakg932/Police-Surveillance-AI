export const detectedPersons = [
  {
    id: 12,
    startTime: '00:02:10',
    endTime: '00:02:25',
    blueShirt: true,
    helmet: true,
    motorcycle: true,
    confidence: 98,
    thumbnail: '/api/placeholder/120/80'
  },
  {
    id: 15,
    startTime: '00:03:15',
    endTime: '00:03:45',
    blueShirt: false,
    helmet: false,
    motorcycle: true,
    confidence: 95,
    thumbnail: '/api/placeholder/120/80'
  },
  {
    id: 18,
    startTime: '00:05:30',
    endTime: '00:06:10',
    blueShirt: true,
    helmet: true,
    motorcycle: false,
    confidence: 92,
    thumbnail: '/api/placeholder/120/80'
  }
];

export const workflowSteps = [
  'Video Upload',
  'Frame Extraction',
  'Object Detection',
  'Attribute Detection',
  'Person Tracking',
  'Query Matching',
  'Results'
];

export const statsData = {
  totalPersons: 24,
  totalMotorcycles: 18,
  withHelmet: 12,
  processingTime: '2.4s'
};

export const reportsData = [
  {
    id: 1,
    title: 'Detection Report - March 15, 2024',
    description: '24 persons, 18 motorcycles, 12 helmets',
    type: 'PDF'
  },
  {
    id: 2,
    title: 'Evidence Package - Case #12345',
    description: '12 video clips, 24 snapshots',
    type: 'ZIP'
  },
  {
    id: 3,
    title: 'Timeline Report - Blue Shirt Persons',
    description: '8 events with timestamps and confidence scores',
    type: 'CSV'
  }
];