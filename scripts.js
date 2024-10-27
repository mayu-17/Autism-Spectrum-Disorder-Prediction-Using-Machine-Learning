document.getElementById('age-form').addEventListener('submit', function(e) {
    e.preventDefault();
    document.getElementById('age-form').classList.add('hidden');
    document.getElementById('questions-form').classList.remove('hidden');

    var age = document.getElementById('age').value;
    var questionsContainer = document.getElementById('questions-container');
    questionsContainer.innerHTML = ''; // Clear existing content

    // Fetch questions based on age group and populate the container
    var questions;
    if (age <= 10) {
        questions = childrenQuestions;
    } else if (age <= 17) {
        questions = adolescentQuestions;
    } else if (age <= 35) {
        questions = youngAdultQuestions;
    } else {
        questions = adultQuestions;
    }

    questions.forEach((question, index) => {
        questionsContainer.innerHTML += `
            <div class="question">
                <label>${index + 1}. ${question}</label>
                <input type="radio" name="Q${index + 1}" value="1" required> Yes
                <input type="radio" name="Q${index + 1}" value="0" required> No
            </div>`;
    });
});
