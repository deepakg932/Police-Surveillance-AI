import React from 'react';
import { FileText, Download } from 'lucide-react';

const ReportsList = ({ reports }) => {


  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700 p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold text-white">Export Reports</h2>
        <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors flex items-center space-x-2">
          <Download className="h-4 w-4" />
          <span>Export All</span>
        </button>
      </div>

      <div className="space-y-4">
        {reports.map((report) => (
          <div key={report.id} className="flex items-center justify-between p-4 bg-gray-700/30 rounded-lg">
            <div className="flex items-center space-x-3">
              <FileText className="h-5 w-5 text-gray-400" />
              <div>
                <p className="text-white font-medium">{report.title}</p>
                <p className="text-sm text-gray-400">{report.description}</p>
              </div>
            </div>
            <button className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm transition-colors">
              Download {report.type}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ReportsList;