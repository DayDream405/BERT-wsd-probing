// Annotation data
let annotationData = [];
let currentIndex = 0;
let selectedOption = null;
let selectedDifficulty = null;
let selectedKeywords = [];
let annotationCount = 0;
let currentFileName = '';
let currentStage = 1;

// DOM elements
const contextElement = document.getElementById('context');
const targetWordElement = document.getElementById('targetWord');
const optionsContainer = document.getElementById('optionsContainer');
const resetBtn = document.getElementById('resetBtn');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const continueBtn = document.getElementById('continueBtn');
const loadDataBtn = document.getElementById('loadDataBtn');
const exportDataBtn = document.getElementById('exportDataBtn');
const resultMessage = document.getElementById('resultMessage');
const annotationCountElement = document.getElementById('annotationCount');
const currentIndexElement = document.getElementById('currentIndex');
const totalItemsElement = document.getElementById('totalItems');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const contextKeywordsElement = document.getElementById('context-keywords');
const selectedKeywordsList = document.getElementById('selectedKeywordsList');
const selectedCountElement = document.getElementById('selectedCount');

// Stage indicators
const stage1Element = document.getElementById('stage1');
const stage2Element = document.getElementById('stage2');
const stage3Element = document.getElementById('stage3');
const stage1Content = document.getElementById('stage1-content');
const stage2Content = document.getElementById('stage2-content');
const stage3Content = document.getElementById('stage3-content');

document.addEventListener('DOMContentLoaded', function() {
    const progress = loadProgress();
    if (progress) {
        fileInfo.textContent = `Unfinished annotation progress detected - File: ${progress.currentFileName} - Last saved: ${new Date(progress.lastSaved).toLocaleString()}`;
        showResult('Unfinished annotation progress detected. Please select the corresponding JSON file to continue.', 'success');
    }
});

// Initialize page
function initializePage() {
    // Set up event listeners
    resetBtn.addEventListener('click', resetSelection);
    prevBtn.addEventListener('click', showPreviousItem);
    nextBtn.addEventListener('click', showNextItem);
    continueBtn.addEventListener('click', continueToNextStage);
    loadDataBtn.addEventListener('click', loadData);
    exportDataBtn.addEventListener('click', exportData);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Set up difficulty option click events
    document.querySelectorAll('.difficulty-option').forEach(option => {
        option.addEventListener('click', function() {
            selectDifficulty(parseInt(this.dataset.level));
        });
    });
    
    // Update annotation count
    updateAnnotationCount();
}

// Load data
function loadData() {
    fileInput.click();
}

// Local storage related functions
// Save progress to local storage
function saveProgress() {
    const progress = {
        annotationData,
        currentIndex,
        currentFileName,
        lastSaved: new Date().toISOString()
    };
    localStorage.setItem('wsd_annotation_progress', JSON.stringify(progress));
    console.log('Progress saved');
}

// Load progress from local storage
function loadProgress() {
    const saved = localStorage.getItem('wsd_annotation_progress');
    if (saved) {
        try {
            const progress = JSON.parse(saved);
            return progress;
        } catch (error) {
            console.error('Failed to load progress:', error);
            return null;
        }
    }
    return null;
}

// Clear local storage progress
function clearProgress() {
    localStorage.removeItem('wsd_annotation_progress');
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    currentFileName = file.name;
    fileInfo.textContent = `Selected file: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);
            
            // Validate data format
            if (!Array.isArray(data)) {
                throw new Error('Data format error: should be JSON array');
            }
            
            // Check if there's saved progress
            const progress = loadProgress();
            if (progress && progress.currentFileName === file.name) {
                // Use saved progress
                if (confirm('Unfinished annotation progress detected. Continue?')) {
                    annotationData = progress.annotationData;
                    currentIndex = progress.currentIndex;
                    showResult(`Progress restored! Last saved: ${new Date(progress.lastSaved).toLocaleString()}`, 'success');
                } else {
                    // Start over
                    annotationData = data.map(item => ({
                        ...item,
                        user_selection: item.user_selection || null,
                        difficulty: item.difficulty || null,
                        keywords: item.keywords || []
                    }));
                    currentIndex = 0;
                    showResult(`Data loaded successfully! Total ${annotationData.length} items`, 'success');
                }
            } else {
                // No progress or file mismatch, start over
                annotationData = data.map(item => ({
                    ...item,
                    user_selection: item.user_selection || null,
                    difficulty: item.difficulty || null,
                    keywords: item.keywords || []
                }));
                currentIndex = 0;
                showResult(`Data loaded successfully! Total ${annotationData.length} items`, 'success');
            }
            
            // Update display
            updateDisplay();
            
        } catch (error) {
            showResult('File parsing error: ' + error.message, 'error');
            console.error('Parsing error:', error);
        }
    };
    reader.onerror = function() {
        showResult('File reading failed', 'error');
    };
    reader.readAsText(file);
}

// Update display
function updateDisplay() {
    if (annotationData.length === 0) {
        contextElement.innerHTML = 'Please click "Select JSON File" to load annotation data';
        targetWordElement.textContent = '-';
        optionsContainer.innerHTML = '';
        document.getElementById('optionsTextDisplay').innerHTML = '';
        currentIndexElement.textContent = '0';
        totalItemsElement.textContent = '0';
        return;
    }
    
    const currentItem = annotationData[currentIndex];
    
    // Set context, highlight target word
    const highlightedContext = currentItem.context.replace(
        new RegExp(currentItem.target, 'g'), 
        `<span class="target-word">${currentItem.target}</span>`
    );
    contextElement.innerHTML = highlightedContext;
    
    // Set target word
    targetWordElement.textContent = currentItem.target;
    
    // Generate options
    generateOptions(currentItem.polysemous.sense_definitions_list);
    
    // If there was a previous selection, restore selection state
    if (currentItem.user_selection !== null) {
        // Find the corresponding option element
        const optionElements = document.querySelectorAll('.option');
        for (let optionElement of optionElements) {
            if (parseInt(optionElement.dataset.originalIndex) === currentItem.user_selection) {
                optionElement.classList.add('selected');
                selectedOption = currentItem.user_selection;
                
                // Highlight corresponding text display
                const displayIndex = optionElement.dataset.displayIndex;
                const selectedTextItem = document.querySelector(`.option-item[data-original-index="${currentItem.user_selection}"]`);
                if (selectedTextItem) {
                    selectedTextItem.style.backgroundColor = '#e1f5fe';
                    selectedTextItem.style.borderColor = '#0288d1';
                }
                break;
            }
        }
    } else {
        resetSelection();
    }
    
    // Update progress
    currentIndexElement.textContent = currentIndex + 1;
    totalItemsElement.textContent = annotationData.length;
    
    // Update button states
    prevBtn.disabled = currentIndex === 0;
    nextBtn.disabled = currentIndex === annotationData.length - 1;
    
    // Reset stages
    resetStages();
}

// Generate options
function generateOptions(options) {
    optionsContainer.innerHTML = '';
    const optionsTextDisplay = document.getElementById('optionsTextDisplay');
    
    // Get correct options for current data
    const currentItem = annotationData[currentIndex];
    const correctOptions = currentItem.correct_definitions_in_context || [];
    
    let finalOptions = [];
    let originalIndices = []; // Save original indices
    
    if (options.length <= 4) {
        // If options are 4 or fewer, use all options
        finalOptions = [...options];
        originalIndices = options.map((_, index) => index);
    } else {
        // If options are more than 4, randomly select 3 wrong options + 1 correct option
        const wrongOptions = options.filter(option => 
            !correctOptions.includes(option)
        );
        
        // Randomly select 3 wrong options
        const randomWrongOptions = shuffleArray([...wrongOptions]).slice(0, 3);
        const wrongIndices = randomWrongOptions.map(option => options.indexOf(option));
        
        // Randomly select one correct option
        const randomCorrectOption = correctOptions.length > 0 
            ? correctOptions[Math.floor(Math.random() * correctOptions.length)]
            : options[0];
        const correctIndex = options.indexOf(randomCorrectOption);
        
        // Combine and shuffle
        const allOptions = [...randomWrongOptions, randomCorrectOption];
        const allIndices = [...wrongIndices, correctIndex];
        
        // Shuffle while maintaining correspondence
        const shuffled = shuffleArray(allOptions.map((option, i) => ({option, index: allIndices[i]})));
        finalOptions = shuffled.map(item => item.option);
        originalIndices = shuffled.map(item => item.index);
    }
    
    // Generate option text display
    let optionsHtml = `
        <div class="options-text-title">
            <i>📝</i> Candidate Sense Text (for translation reference)
        </div>
    `;
    
    finalOptions.forEach((option, displayIndex) => {
        const originalIndex = originalIndices[displayIndex];
        
        optionsHtml += `
            <div class="option-item" data-original-index="${originalIndex}">
                <span class="option-number-circle">${displayIndex + 1}</span>
                <span class="option-text-content">${option}</span>
            </div>
        `;
    });
    
    optionsTextDisplay.innerHTML = optionsHtml;
    
    // Generate option buttons
    finalOptions.forEach((option, displayIndex) => {
        const originalIndex = originalIndices[displayIndex];
        const optionElement = document.createElement('div');
        optionElement.className = 'option';
        optionElement.dataset.originalIndex = originalIndex;
        optionElement.dataset.displayIndex = displayIndex;
        
        optionElement.innerHTML = `
            <div class="option-number">${displayIndex + 1}</div>
            <div class="option-text">${option.substring(0, 60)}${option.length > 60 ? '...' : ''}</div>
        `;
        
        optionElement.addEventListener('click', () => selectOption(optionElement, originalIndex, displayIndex));
        
        optionsContainer.appendChild(optionElement);
    });
}

// Array shuffle function
function shuffleArray(array) {
    const newArray = [...array];
    for (let i = newArray.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
    }
    return newArray;
}

// Select option
function selectOption(optionElement, originalIndex, displayIndex) {
    // Remove previous selection
    const previouslySelected = document.querySelector('.option.selected');
    if (previouslySelected) {
        previouslySelected.classList.remove('selected');
    }
    
    // Set current selection
    optionElement.classList.add('selected');
    selectedOption = originalIndex;
    
    // Highlight corresponding option in text display
    const textItems = document.querySelectorAll('.option-item');
    textItems.forEach(item => {
        item.style.backgroundColor = '#fff';
        item.style.borderColor = '#e0e0e0';
    });
    
    if (displayIndex !== undefined) {
        const selectedTextItem = document.querySelector(`.option-item[data-original-index="${originalIndex}"]`);
        if (selectedTextItem) {
            selectedTextItem.style.backgroundColor = '#e1f5fe';
            selectedTextItem.style.borderColor = '#0288d1';
        }
    }
    
    // Save user selection
    annotationData[currentIndex].user_selection = originalIndex;
    
    // Update annotation count
    updateAnnotationCount();
    
    // Save progress
    saveProgress();
    
    // Automatically proceed to next stage
    setTimeout(() => {
        continueToNextStage();
    }, 500);
}

// Select difficulty
function selectDifficulty(level) {
    // Remove previous selection
    document.querySelectorAll('.difficulty-option').forEach(option => {
        option.classList.remove('selected');
    });
    
    // Set current selection
    document.querySelector(`.difficulty-option[data-level="${level}"]`).classList.add('selected');
    selectedDifficulty = level;
    
    // Save difficulty selection
    annotationData[currentIndex].difficulty = level;
    
    // Save progress
    saveProgress();
    
    // Automatically proceed to next stage
    setTimeout(() => {
        continueToNextStage();
    }, 500);
}

// Select keyword
function selectKeyword(word) {
    // If already selected, deselect
    if (selectedKeywords.includes(word)) {
        selectedKeywords = selectedKeywords.filter(w => w !== word);
        updateSelectedKeywordsDisplay();
        
        // Save progress
        saveProgress();
        return;
    }
    
    // If less than 3, add
    if (selectedKeywords.length < 3) {
        selectedKeywords.push(word);
        updateSelectedKeywordsDisplay();
        
        // Save keyword selection
        annotationData[currentIndex].keywords = [...selectedKeywords];
        
        // Save progress
        saveProgress();
        
        // If 3 keywords selected, automatically proceed
        if (selectedKeywords.length === 3) {
            setTimeout(() => {
                continueToNextStage();
            }, 500);
        }
    } else {
        showResult('Maximum 3 keywords allowed', 'error');
    }
}

// Update selected keywords display
function updateSelectedKeywordsDisplay() {
    selectedKeywordsList.innerHTML = '';
    selectedCountElement.textContent = selectedKeywords.length;
    
    selectedKeywords.forEach(word => {
        const keywordTag = document.createElement('span');
        keywordTag.className = 'keyword-tag';
        keywordTag.innerHTML = `${word} <span class="remove" data-word="${word}">×</span>`;
        selectedKeywordsList.appendChild(keywordTag);
    });
    
    // Add events for remove buttons
    document.querySelectorAll('.keyword-tag .remove').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const word = this.getAttribute('data-word');
            removeKeyword(word);
        });
    });
}

// Remove keyword
function removeKeyword(word) {
    selectedKeywords = selectedKeywords.filter(w => w !== word);
    updateSelectedKeywordsDisplay();
    
    // Save keyword selection
    annotationData[currentIndex].keywords = [...selectedKeywords];
    
    // Save progress
    saveProgress();
    
    // Update selection state in context
    updateContextKeywordsSelection();
}

// Update keyword selection state in context
function updateContextKeywordsSelection() {
    const words = contextKeywordsElement.querySelectorAll('.context-word');
    words.forEach(wordElement => {
        const word = wordElement.textContent.trim();
        if (selectedKeywords.includes(word)) {
            wordElement.classList.add('selected');
        } else {
            wordElement.classList.remove('selected');
        }
    });
}

// Reset selection
function resetSelection() {
    if (currentStage === 1) {
        const selected = document.querySelector('.option.selected');
        if (selected) {
            selected.classList.remove('selected');
        }
        
        // Reset text display highlighting
        const textItems = document.querySelectorAll('.option-item');
        textItems.forEach(item => {
            item.style.backgroundColor = '#fff';
            item.style.borderColor = '#e0e0e0';
        });
        
        selectedOption = null;
        annotationData[currentIndex].user_selection = null;
    } else if (currentStage === 2) {
        document.querySelectorAll('.difficulty-option').forEach(option => {
            option.classList.remove('selected');
        });
        selectedDifficulty = null;
        annotationData[currentIndex].difficulty = null;
    } else if (currentStage === 3) {
        selectedKeywords = [];
        annotationData[currentIndex].keywords = [];
        updateSelectedKeywordsDisplay();
        updateContextKeywordsSelection();
    }
    
    resultMessage.className = 'result';
    resultMessage.style.display = 'none';
    
    // Update annotation count
    updateAnnotationCount();
    
    // Save progress
    saveProgress();
}

// Continue to next stage
function continueToNextStage() {
    if (currentStage === 1 && selectedOption === null) {
        showResult('Please select a sense option first', 'error');
        return;
    }
    
    if (currentStage === 2 && selectedDifficulty === null) {
        showResult('Please select a difficulty level first', 'error');
        return;
    }
    
    if (currentStage === 3 && selectedKeywords.length < 3) {
        showResult('Please select 3 keywords', 'error');
        return;
    }
    
    // Move to next stage
    currentStage++;
    
    if (currentStage > 3) {
        // Complete current item, move to next
        if (currentIndex < annotationData.length - 1) {
            currentIndex++;
            currentStage = 1;
            
            // Save progress
            saveProgress();
            
            updateDisplay();
            showResult('Annotation complete! Automatically moving to next item', 'success');
        } else {
            // All data annotated, clear progress
            clearProgress();
            showResult('All data annotated! Progress cleared', 'success');
            currentStage = 1;
            updateDisplay();
        }
    } else {
        updateStageDisplay();
        
        // Save progress
        saveProgress();
    }
}

// Update stage display
function updateStageDisplay() {
    // Update stage indicators
    stage1Element.classList.toggle('active', currentStage === 1);
    stage2Element.classList.toggle('active', currentStage === 2);
    stage3Element.classList.toggle('active', currentStage === 3);
    
    // Update content display
    stage1Content.style.display = currentStage === 1 ? 'block' : 'none';
    stage2Content.style.display = currentStage === 2 ? 'block' : 'none';
    stage3Content.style.display = currentStage === 3 ? 'block' : 'none';
    
    // If stage 3, create clickable context
    if (currentStage === 3) {
        createClickableContext();
        selectedKeywords = annotationData[currentIndex].keywords || [];
        updateSelectedKeywordsDisplay();
    }
    
    // If stage 2, restore difficulty selection for current data
    if (currentStage === 2) {
        // First clear all selections
        document.querySelectorAll('.difficulty-option').forEach(option => {
            option.classList.remove('selected');
        });
        
        // Only restore difficulty selection for current data
        selectedDifficulty = annotationData[currentIndex].difficulty || null;
        if (selectedDifficulty) {
            document.querySelector(`.difficulty-option[data-level="${selectedDifficulty}"]`).classList.add('selected');
        }
    }
}

// Create clickable context
function createClickableContext() {
    const currentItem = annotationData[currentIndex];
    // Use more precise tokenization, preserving punctuation
    const words = currentItem.context.split(/(\s+|[.,!?;:])/).filter(word => word.trim() !== '');
    
    contextKeywordsElement.innerHTML = '';
    
    words.forEach(word => {
        if (word.trim() === '') {
            contextKeywordsElement.innerHTML += word;
        } else {
            const isTarget = word === currentItem.target;
            const isSelected = selectedKeywords.includes(word);
            const wordElement = document.createElement('span');
            wordElement.className = `context-word ${isTarget ? 'target-word' : ''} ${isSelected ? 'selected' : ''}`;
            wordElement.textContent = word;
            
            if (!isTarget && !/[.,!?;:\s]/.test(word)) {
                wordElement.style.cursor = 'pointer';
                wordElement.addEventListener('click', () => selectKeyword(word));
            } else {
                wordElement.style.cursor = 'default';
            }
            
            contextKeywordsElement.appendChild(wordElement);
            
            // Add space (except for punctuation)
            if (!/[.,!?;:]$/.test(word)) {
                contextKeywordsElement.appendChild(document.createTextNode(' '));
            }
        }
    });
}

// Reset stages
function resetStages() {
    currentStage = 1;
    selectedDifficulty = null;
    selectedKeywords = [];
    updateStageDisplay();
}

// Show previous item
function showPreviousItem() {
    if (currentIndex > 0) {
        currentIndex--;
        updateDisplay();
    }
}

// Show next item
function showNextItem() {
    if (currentIndex < annotationData.length - 1) {
        currentIndex++;
        updateDisplay();
    }
}

// Export data
function exportData() {
    if (annotationData.length === 0) {
        showResult('No data to export', 'error');
        return;
    }
    
    // Create data copy
    const exportData = JSON.parse(JSON.stringify(annotationData));
    
    // Create Blob object
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    
    // Create download link
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = currentFileName ? `annotated_${currentFileName}` : 'annotated_data.json';
    document.body.appendChild(a);
    a.click();
    
    // Cleanup
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 0);
    
    const statsParam = `?total=${annotationCount}&time=${Date.now()}`;
    
    // Ask whether to clear progress
    if (confirm('Data exported successfully! Clear current annotation progress?')) {
        clearProgress();
        showResult('Data exported successfully! Progress cleared', 'success');
        // location.href = "thanks.html" + statsParam;
    } else {
        showResult('Data exported successfully! Progress saved', 'success');
    }
}

// Show result message
function showResult(message, type) {
    resultMessage.textContent = message;
    resultMessage.className = `result ${type}`;
    resultMessage.style.display = 'block';
    
    // Automatically hide success messages after 3 seconds
    if (type === 'success') {
        setTimeout(() => {
            resultMessage.style.display = 'none';
        }, 3000);
    }
}

// Update annotation count
function updateAnnotationCount() {
    if (annotationData.length === 0) {
        annotationCount = 0;
    } else {
        annotationCount = annotationData.filter(item => 
            item.user_selection !== null && 
            item.difficulty !== null && 
            item.keywords.length > 0 && item.keywords.length <= 3
        ).length;
    }
    annotationCountElement.textContent = annotationCount;
}

// Initialize page
initializePage();