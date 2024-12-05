import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import USFlag from './flags/us-flag.svg';
import IndiaFlag from './flags/india-flag-image.png';
import logo1 from './mudra-ai-logos/logo1.jpeg';

import { Hands, HAND_CONNECTIONS } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';

const App = () => {
  const [mode, setMode] = useState('training'); // 'training' or 'testing'
  const [language, setLanguage] = useState('asl'); // 'asl' or 'isl'
  const [currentGesture, setCurrentGesture] = useState(null);
  const [feedback, setFeedback] = useState('');
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    requestGesture();
  }, [language, mode]);

  useEffect(() => {
    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      },
    });

    // Adjust maxNumHands based on language
    hands.setOptions({
      maxNumHands: language === 'isl' ? 2 : 1, // ISL uses two hands
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    hands.onResults(onResults);

    const videoElement = videoRef.current;
    if (videoElement) {
      const camera = new Camera(videoElement, {
        onFrame: async () => {
          await hands.send({ image: videoElement });
        },
        width: 1280,
        height: 720,
      });
      camera.start();

      // Clean up on unmount or language change
      return () => {
        camera.stop();
        hands.close();
      };
    }
  }, [language]); // Reinitialize when language changes

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const onResults = (results) => {
    const canvasElement = canvasRef.current;
    const canvasCtx = canvasElement.getContext('2d');

    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;

    canvasElement.width = videoWidth;
    canvasElement.height = videoHeight;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Draw the video frame
    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    // Draw the landmarks
    if (results.multiHandLandmarks && results.multiHandedness) {
      for (let index = 0; index < results.multiHandLandmarks.length; index++) {
        const classification = results.multiHandedness[index];
        const isRightHand = classification.label === 'Right';
        const landmarks = results.multiHandLandmarks[index];
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
          color: isRightHand ? '#00FF00' : '#FF0000',
          lineWidth: 5,
        });
        drawLandmarks(canvasCtx, landmarks, {
          color: isRightHand ? '#00FF00' : '#FF0000',
          lineWidth: 2,
        });
      }
    }
    canvasCtx.restore();
  };

  const requestGesture = () => {
    const gestures =
      language === 'asl'
        ? [
            'A',
            'B',
            'C',
            'D',
            'E',
            'F',
            'G',
            'H',
            'I',
            'J',
            'K',
            'L',
            'M',
            'N',
            'O',
            'P',
            'Q',
            'R',
            'S',
            'T',
            'U',
            'V',
            'W',
            'X',
            'Y',
            'Z',
          ]
        : [
            // Update with actual ISL gestures
            'A',
            'B',
            'D', 'E', 'F', 'G', 'H'
            // ... add more ISL gestures
          ];
    const randomGesture =
      gestures[Math.floor(Math.random() * gestures.length)];
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
          const formData = new FormData();
          formData.append('image', blob, 'gesture.jpg');
          formData.append('expected_sign', currentGesture);
          formData.append('language', language);

          fetch('https://web-production-08f6.up.railway.app/predict', {
            method: 'POST',
            body: formData,
          })
            .then((response) => response.json())
            .then((data) => {
              if (data.result) {
                setFeedback('✅ Correct sign detected!');
              } else {
                setFeedback(
                  `❌ Incorrect sign. Predicted: ${data.predicted_sign}`
                );
              }
              requestGesture();
            })
            .catch((error) => {
              console.error('Error:', error);
              setFeedback('Error occurred while checking gesture.');
            });
        },
        'image/jpeg',
        0.95
      );
    } else {
      setFeedback('Video not ready. Please try again.');
    }
  };

  // Automatically check gesture after 1 second for ISL
  useEffect(() => {
    if (language === 'isl') {
      const timer = setTimeout(() => {
        checkGesture();
        // requestGesture();
      }, 5000); // 1-second delay

      return () => clearTimeout(timer); // Clean up the timer on unmount or change
    }
  }, [currentGesture, language]);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <div className="App">
      <div className="theme-toggle">
        <div 
          className={`theme-toggle-track ${theme}`} 
          onClick={toggleTheme}
        >
          <div className="theme-toggle-thumb">
            {theme === 'light' ? (
              <span className="theme-icon">☀️</span>
            ) : (
              <span className="theme-icon">🌙</span>
            )}
          </div>
        </div>
      </div>

      <div className="header">
        <img
          src={logo1}
          className="logo"
          style={{ borderRadius: '50%' }}
          alt="Logo"
        />
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
          <img src={USFlag} className="button-icon" alt="US Flag" />
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
            alt="India Flag"
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
                mode === 'training' ? (
                  <img
                    className="gesture-image"
                    src={`/${language}_dataset/${currentGesture}.png`} // Updated to use language
                    alt={`Hand sign for ${currentGesture}`}
                  />
                ) : (
                  <img
                    className="gesture-image"
                    src={`/alphabet_images/${currentGesture}.png`}
                    alt={`Letter ${currentGesture}`}
                  />
                )
              ) : (
                <p>No gesture selected.</p>
              )}
            </div>
          </div>

          <div className="camera-feed">
            <div className="gesture-info">
              <h2>Make Gesture for: {currentGesture}</h2>
              <div className="video-container">
                <video
                  ref={videoRef}
                  className="video-feed"
                  autoPlay
                  playsInline
                ></video>
                <canvas ref={canvasRef} className="canvas-overlay"></canvas>
              </div>
            </div>
          </div>
        </div>
        <button
          className="check-gesture-button"
          onClick={checkGesture}
          disabled={language === 'isl'} // Optionally disable for ISL
        >
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
