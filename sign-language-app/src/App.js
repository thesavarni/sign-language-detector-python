import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import USFlag from './flags/us-flag.svg'; // Import the SVG file
import IndiaFlag from './flags/india-flag-image.png'; // Import the SVG file
import logo1 from './mudra-ai-logos/logo1.jpeg'



const App = () => {
  const [mode, setMode] = useState('training'); // 'training' or 'testing'
  const [language, setLanguage] = useState('asl'); // 'asl' or 'isl'
  const [currentGesture, setCurrentGesture] = useState(null);
  const [feedback, setFeedback] = useState('');
  const videoRef = useRef(null);

  useEffect(() => {
    requestGesture();
    startVideoStream();
  }, [language]);

  const startVideoStream = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
      })
      .catch((err) => {
        console.error('Error accessing webcam: ', err);
      });
  };

  const requestGesture = () => {
    const gestures = language === 'asl'
      ? ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z']
      : ['X', 'Y', 'Z']; // Update with ISL gestures if available
    const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
    setCurrentGesture(randomGesture);
  };

  const checkGesture = () => {
    setFeedback('Checking...');

    const canvas = document.createElement('canvas');
    const video = videoRef.current;

    if (video && video.videoWidth > 0 && video.videoHeight > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          // Create a FormData object to send image and expected_sign
          const formData = new FormData();
          formData.append('image', blob, 'gesture.jpg');
          formData.append('expected_sign', currentGesture);
          formData.append('language', language)

          fetch('https://web-production-08f6.up.railway.app/predict', {
            method: 'POST',
            body: formData,
          })
            .then((response) => response.json())
            .then((data) => {
              if (data.result) {
                setFeedback('✅ Correct sign detected!');
              } else {
                setFeedback(`❌ Incorrect sign. Predicted: ${data.predicted_sign}`);
              }
              // Request a new gesture for the next round
              requestGesture();
            })
            .catch((error) => {
              console.error('Error:', error);
              setFeedback('Error occurred while checking gesture.');
            });
        },
        'image/jpeg',
        0.95 // Quality parameter (optional)
      );
    } else {
      setFeedback('Video not ready. Please try again.');
    }
  };

  return (
    <div className="App">
      <div className="header">
      <img src={logo1} className="logo"  style={{ borderRadius: '50%' }}/>
        <h1>MUDRA AI</h1>
      </div>

      <div className="language-toggle">
        <button
          className={language === 'asl' ? 'active' : ''}
          onClick={() => {
            setLanguage('asl');
            setFeedback('');
          }}
        >
          <img src={USFlag} className="button-icon" />
          <span>ASL</span>
        </button>
        <button
          className={language === 'isl' ? 'active' : ''}
          onClick={() => {
            setLanguage('isl');
            setFeedback('');
          }}
        >
        <img 
          src={IndiaFlag}
          style={{ width: '40px', height: '40px' }}
          className="button-icon" 
        />
          <span>ISL</span>
        </button>
      </div>

      <div className="content-platform">
        <div className="content">
          <div className="gesture-display">
            <div className="gesture-info">
              <h2>Gesture: {currentGesture}</h2>
              {currentGesture ? (
                <img
                  className="gesture-image"
                  src={`/asl_dataset/${currentGesture}.png`}
                  alt={`Hand sign for ${currentGesture}`}
                />
              ) : (
                <p>No gesture selected.</p>
              )}
            </div>
          </div>

          <div className="camera-feed">
            <div className="gesture-info">
              <h2>Make Gesture for: {currentGesture}</h2>
              <video ref={videoRef} autoPlay className="video-feed"></video>
            </div>
          </div>
        </div>
        <button className="check-gesture-button" onClick={checkGesture}>
          Check Gesture
        </button>
      </div>

      <div className="mode-toggle">
        <button
          className={mode === 'training' ? 'active' : ''}
          onClick={() => {
            setMode('training');
            setFeedback('');
          }}
        >
          Training Mode
        </button>
        <button
          className={mode === 'testing' ? 'active' : ''}
          onClick={() => {
            setMode('testing');
            setFeedback('');
          }}
        >
          Testing Mode
        </button>
      </div>

      <div className="feedback">
        <h2>{feedback}</h2>
      </div>
    </div>
  );
};

export default App;