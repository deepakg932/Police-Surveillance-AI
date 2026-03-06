import React from 'react';
import { CheckCircle } from 'lucide-react';

const WorkflowProgress = ({ steps, currentStep = 3 }) => {
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-4">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => (
          <React.Fragment key={step}>
            <div className="flex flex-col items-center">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                index < currentStep ? 'bg-green-500' : 
                index === currentStep ? 'bg-blue-500 animate-pulse' : 'bg-gray-700'
              }`}>
                {index < currentStep ? <CheckCircle className="h-4 w-4 text-white" /> : index + 1}
              </div>
              <span className="text-xs text-gray-400 mt-1">{step}</span>
            </div>
            {index < steps.length - 1 && <div className="w-12 h-0.5 bg-gray-700 mx-2"></div>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default WorkflowProgress;