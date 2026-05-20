document.addEventListener('DOMContentLoaded', () => {
    // Buttons
    const navStartBtn = document.getElementById('nav-start-btn');
    const heroStartBtn = document.getElementById('hero-start-btn');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const stopCameraBtn = document.getElementById('stop-camera-btn');

    // Video Elements
    const videoFeed = document.getElementById('webcam-feed');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const liveIndicator = document.getElementById('live-indicator');

    // Detection UI
    const detectedWordEl = document.getElementById('detected-word');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');

    let stream = null;
    let detectionInterval = null;

    // Words array that will be populated from the actual SQLite database
    let databaseWords = [];

    // Live Sentence UI Variables
    const liveSentenceText = document.getElementById('live-sentence-text');
    const clearSentenceBtn = document.getElementById('clear-sentence-btn');
    let sentenceArray = [];
    let lastDetectedWord = "";

    if (clearSentenceBtn) {
        clearSentenceBtn.addEventListener('click', () => {
            sentenceArray = [];
            // Leaving lastDetectedWord untouched ensures an ongoing sign isn't immediately re-added!
            liveSentenceText.innerHTML = `<span class="opacity-50 italic text-base">Waiting for gestures...</span>`;
            if (document.getElementById('ai-sentence-result-box')) {
                document.getElementById('ai-sentence-result-box').classList.add('hidden');
            }
        });
    }

    // Function to start the camera
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoFeed.srcObject = stream;
            
            // Toggle Display
            videoPlaceholder.classList.add('hidden');
            videoFeed.classList.remove('hidden');
            liveIndicator.classList.remove('hidden');

            // Button states
            startCameraBtn.disabled = true;
            startCameraBtn.classList.add('opacity-50', 'cursor-not-allowed');
            stopCameraBtn.disabled = false;
            stopCameraBtn.classList.remove('opacity-50', 'cursor-not-allowed');

            // Scroll to detection section
            document.getElementById('detection').scrollIntoView({ behavior: 'smooth' });

            // Start real-time detection pipeline
            startRealtimeDetection();

        } catch (err) {
            console.error("Error accessing the camera: ", err);
            alert("Could not access the camera. Please ensure you have granted permission.");
        }
    }

    // Function to stop the camera
    function stopCamera() {
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            videoFeed.srcObject = null;
            stream = null;
        }

        // Toggle Display
        videoPlaceholder.classList.remove('hidden');
        videoFeed.classList.add('hidden');
        liveIndicator.classList.add('hidden');

        // Button states
        startCameraBtn.disabled = false;
        startCameraBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        stopCameraBtn.disabled = true;
        stopCameraBtn.classList.add('opacity-50', 'cursor-not-allowed');

        // Stop real-time detection
        stopRealtimeDetection();
        
        // Reset Detection UI
        detectedWordEl.textContent = "WAITING...";
        confidenceBar.style.width = "0%";
        confidenceText.textContent = "0%";
    }

    const hiddenCanvas = document.createElement('canvas');
    const hiddenCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

    async function startRealtimeDetection() {
        if (!stream) return;
        
        // Wait for video meta data to ensure dimensions are correct
        if (videoFeed.videoWidth === 0) {
            await new Promise(resolve => videoFeed.onloadedmetadata = resolve);
        }
        
        hiddenCanvas.width = videoFeed.videoWidth || 640;
        hiddenCanvas.height = videoFeed.videoHeight || 480;

        detectionInterval = setInterval(async () => {
             // 1. Draw web browser frame to hidden canvas
             hiddenCtx.drawImage(videoFeed, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
             
             // 2. Compress frame to JPEG
             const imageB64 = hiddenCanvas.toDataURL('image/jpeg', 0.8);
             
             // 3. Send to Flask Backend SignDetector via HTTP
             try {
                 const response = await fetch('/api/detect', {
                     method: 'POST',
                     headers: { 'Content-Type': 'application/json' },
                     body: JSON.stringify({ image: imageB64 })
                 });
                 
                 const data = await response.json();
                 if (data.success) {
                    const rawWord = data.word; // e.g. "Waiting...", "HELLO", "Train Model First"
                    const conf = Math.floor(data.confidence);

                    if (rawWord === "Train Model First") {
                        detectedWordEl.textContent = "TRAIN MODEL FIRST";
                        confidenceBar.style.width = "0%";
                        confidenceText.textContent = "0%";
                    } else if (rawWord !== "Waiting...") {
                        detectedWordEl.textContent = rawWord;
                        confidenceBar.style.width = conf + "%";
                        confidenceText.textContent = conf + "%";
                        
                        // Live Sentence Accumulation Logic
                        if (!rawWord.startsWith("UNCLEAR") && rawWord !== lastDetectedWord && rawWord !== "RECORD SIGNS FIRST" && rawWord !== "SERVER ERROR") {
                            sentenceArray.push(rawWord);
                            lastDetectedWord = rawWord;
                            
                            const formattedSentence = sentenceArray.map(w => `<span class="bg-surface-container-highest border border-outline-variant/30 text-primary px-3 py-1.5 rounded-lg mx-1 mt-2 inline-block shadow-sm align-middle font-bold">${w}</span>`).join(" ");
                            liveSentenceText.innerHTML = formattedSentence + "<span class='animate-pulse inline-block w-3 h-6 bg-secondary align-middle ml-2 mt-2 rounded-[2px] opacity-70'></span>";
                            liveSentenceText.parentElement.scrollTop = liveSentenceText.parentElement.scrollHeight;
                        }
                    } else {
                        // "Waiting..." UI logic
                        detectedWordEl.textContent = "WAITING...";
                        confidenceBar.style.width = "0%";
                        confidenceText.textContent = "0%";
                    }
                 }
             } catch (err) {
                 console.error("Detection payload failed:", err);
             }
        }, 40); // Frame sampling rate 40ms -> ~25 fps (matches video recording speed)
    }

    function stopRealtimeDetection() {
        if (detectionInterval) {
            clearInterval(detectionInterval);
            detectionInterval = null;
        }
    }

    // Event Listeners
    navStartBtn.addEventListener('click', (e) => {
        e.preventDefault();
        startCamera();
    });

    // Add Gesture Logic
    const gestureWordInput = document.getElementById('gesture-word-input');
    const startRecordBtn = document.getElementById('start-record-btn');
    const recordBtnText = document.getElementById('record-btn-text');
    const gestureVideoFeed = document.getElementById('gesture-video-feed');
    const gestureVideoPlaceholder = document.getElementById('gesture-video-placeholder');
    const gestureOverlay = document.getElementById('gesture-overlay');
    const recordingIndicator = document.getElementById('recording-indicator');

    let mediaRecorder;
    let recordedChunks = [];

    if (startRecordBtn) {
        startRecordBtn.addEventListener('click', async () => {
            const word = gestureWordInput.value.trim();
            if (!word) {
                alert('Please enter a word or phrase before recording!');
                return;
            }

            try {
                // Request camera
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                gestureVideoFeed.srcObject = stream;
                
                // Initial UI Setup to show video
                gestureVideoPlaceholder.classList.add('hidden');
                gestureOverlay.classList.add('hidden');
                gestureVideoFeed.classList.remove('hidden');
                
                // Wait briefly for camera to warm up
                recordBtnText.textContent = "Get ready...";
                startRecordBtn.disabled = true;
                await new Promise(r => setTimeout(r, 1500));
                
                // Real UI Updates for recording
                recordingIndicator.classList.remove('hidden');
                recordingIndicator.classList.add('flex');
                
                // Start Recording
                recordedChunks = [];
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    // UI Reset
                    recordingIndicator.classList.add('hidden');
                    recordingIndicator.classList.remove('flex');
                    recordBtnText.textContent = "Processing & Saving...";
                    
                    // Construct the blob
                    const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
                    const formData = new FormData();
                    formData.append('video', videoBlob, 'gesture.webm');
                    formData.append('word', word);

                    try {
                        const response = await fetch('/api/add_gesture', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        if (data.success) {
                            alert(data.message);
                            gestureWordInput.value = ''; // clear input
                            if (window.loadGestureLibrary) window.loadGestureLibrary();
                        } else {
                            alert("Error: " + data.error);
                        }
                    } catch (err) {
                        console.error("Upload error:", err);
                        alert("Failed to upload the gesture. Make sure you are running 'python server.py'");
                    }

                    // Final UI Reset
                    stream.getTracks().forEach(track => track.stop());
                    gestureVideoFeed.classList.add('hidden');
                    gestureVideoPlaceholder.classList.remove('hidden');
                    gestureOverlay.classList.remove('hidden');
                    startRecordBtn.disabled = false;
                    recordBtnText.textContent = "Start Recording (5s)";
                };

                mediaRecorder.start();

                // Record exactly 5 seconds
                let secondsLeft = 5;
                recordBtnText.textContent = `Recording... (${secondsLeft}s left)`;
                
                const countdownInterval = setInterval(() => {
                    secondsLeft--;
                    if (secondsLeft > 0) {
                        recordBtnText.textContent = `Recording... (${secondsLeft}s left)`;
                    } else {
                        clearInterval(countdownInterval);
                        mediaRecorder.stop();
                    }
                }, 1000);

            } catch (err) {
                console.error(err);
                alert('Could not access the camera for recording.');
                startRecordBtn.disabled = false;
                recordBtnText.textContent = "Start Recording (5s)";
            }
        });
    }

    heroStartBtn.addEventListener('click', startCamera);
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);

    // AI Sentence Generation
    const generateAiSentenceBtn = document.getElementById('generate-ai-sentence-btn');
    const geminiApiKeyInput = document.getElementById('gemini-api-key');
    const aiSentenceResultBox = document.getElementById('ai-sentence-result-box');
    const aiSentenceResult = document.getElementById('ai-sentence-result');

    if (generateAiSentenceBtn) {
        generateAiSentenceBtn.addEventListener('click', async () => {
            const apiKey = geminiApiKeyInput.value.trim();
            if (sentenceArray.length === 0) {
                alert("No words to translate! Sign something to the camera first.");
                return;
            }

            // Show loading UI
            aiSentenceResultBox.classList.remove('hidden');
            aiSentenceResult.textContent = "Generating...";
            generateAiSentenceBtn.disabled = true;
            generateAiSentenceBtn.classList.add("opacity-50");

            try {
                const res = await fetch('/api/generate_sentence', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ api_key: apiKey, words: sentenceArray })
                });

                const data = await res.json();
                if (data.success) {
                    aiSentenceResult.textContent = data.sentence;
                    
                    // Call browser speech synthesis so they can hear it
                    if ('speechSynthesis' in window) {
                        const utterance = new SpeechSynthesisUtterance(data.sentence);
                        window.speechSynthesis.speak(utterance);
                    }
                } else {
                    aiSentenceResult.textContent = "Error: " + data.error;
                }
            } catch (err) {
                aiSentenceResult.textContent = "Failed to connect to server.";
                console.error(err);
            } finally {
                generateAiSentenceBtn.disabled = false;
                generateAiSentenceBtn.classList.remove("opacity-50");
            }
        });
    }

    // Dynamic Gesture Library Logic
    const gestureLibraryContainer = document.getElementById('gesture-library-container');
    const gestureSearchInput = document.getElementById('gesture-search-input');
    
    let fullLibrary = [];

    window.loadGestureLibrary = async function() {
        if (!gestureLibraryContainer) return;
        
        try {
            const res = await fetch('/api/library');
            const data = await res.json();
            if (data.success) {
                fullLibrary = data.library;
                renderLibrary(fullLibrary);
            }
        } catch (e) {
            console.error("Failed to fetch library", e);
        }
    }

    function renderLibrary(libraryToShow) {
        if (!gestureLibraryContainer) return;
        
        gestureLibraryContainer.innerHTML = ''; // clear

        if (libraryToShow.length === 0) {
            gestureLibraryContainer.innerHTML = '<div class="flex items-center justify-center h-full opacity-50 italic">No gestures found.</div>';
            return;
        }

        libraryToShow.forEach(item => {
            // Icon assigner based on text or random for visual variety
            let iconText = "sign_language";
            if(item.word === "HELLO") iconText = "waving_hand";
            else if(item.word === "THANK YOU") iconText = "front_hand";
            else if(item.word === "PLEASE") iconText = "hail";
            else if(item.word === "FRIEND") iconText = "handshake";
            else if(item.word === "YES") iconText = "thumb_up";

            const card = document.createElement('div');
            card.className = "bg-surface-container-highest rounded-xl p-4 flex items-center justify-between border border-outline-variant/20 hover:border-primary/50 transition-all group";
            card.innerHTML = `
                <div class="flex items-center gap-6">
                    <div class="w-16 h-16 rounded-lg bg-surface-container-lowest overflow-hidden shrink-0 flex items-center justify-center text-primary">
                       <span class="material-symbols-outlined text-3xl">${iconText}</span>
                    </div>
                    <div>
                        <h4 class="text-xl font-bold text-on-surface">${item.word}</h4>
                        <p class="text-sm text-on-surface-variant mt-1">Recorded Gesture</p>
                        <span class="inline-block mt-2 text-xs font-bold text-secondary bg-secondary/10 px-2 py-1 rounded">${item.count} Contributor${item.count > 1 ? 's' : ''}</span>
                    </div>
                </div>
                <button onclick="deleteGestureWord('${item.word}')" class="text-on-surface-variant hover:text-error opacity-0 group-hover:opacity-100 transition-opacity p-2 rounded-full hover:bg-error/10 mr-2 border border-transparent hover:border-error/30" title="Delete Word">
                    <span class="material-symbols-outlined">delete_forever</span>
                </button>
            `;
            gestureLibraryContainer.appendChild(card);
        });
    }

    window.deleteGestureWord = async function(word) {
        if(confirm(`Are you sure you want to completely delete the gesture '${word}' and all its videos?`)) {
            try {
                const res = await fetch('/api/delete_word', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ word })
                });
                const data = await res.json();
                if(data.success) {
                    window.loadGestureLibrary(); // refresh library
                } else {
                    alert('Error deleting: ' + data.error);
                }
            } catch(e) {
                console.error(e);
                alert('Could not delete word.');
            }
        }
    };

    if (gestureSearchInput && gestureLibraryContainer) {
        gestureSearchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            const filtered = fullLibrary.filter(item => item.word.toLowerCase().includes(query));
            renderLibrary(filtered);
        });
        
        // Initial load
        window.loadGestureLibrary();
    }
});
