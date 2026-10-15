import React, { useState } from "react";
import SettingsPanel from "../SettingsPanel/SettingsPanel";
import RightPanel from "../RightPanel/RightPanel";
import "../../assets/css/Body.css";

// Left panel holds the pipeline settings (the inference video is shown in a
// separate OpenCV pop-up window, not in the browser). Right panel holds the
// live metrics. The arrow collapses the metrics panel.
const Body: React.FC = () => {
  const [isRightPanelCollapsed, setIsRightPanelCollapsed] = useState(false);

  return (
    <div className="container">
      <div className={`left-panel${isRightPanelCollapsed ? ' left-panel--full' : ''}`}>
        <SettingsPanel />
      </div>

      {!isRightPanelCollapsed && (
        <div className="right-panel">
          <RightPanel />
        </div>
      )}

      <div
        className={`arrow${isRightPanelCollapsed ? ' collapsed' : ''}`}
        onClick={() => setIsRightPanelCollapsed(prev => !prev)}
        title={isRightPanelCollapsed ? 'Show metrics' : 'Hide metrics'}
      >
        {isRightPanelCollapsed ? '◀' : '▶'}
      </div>
    </div>
  );
};

export default Body;
