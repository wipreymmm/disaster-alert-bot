document.addEventListener('DOMContentLoaded', function() {
    function updateTime() {
        const now = new Date();

        // Format time
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        const seconds = now.getSeconds().toString().padStart(2, '0');

        // Update element
        const timeElement = document.getElementById('current-time');
        if (timeElement) timeElement.textContent = `${hours}:${minutes}:${seconds}`;
    }

    // Run
    updateTime();
    setInterval(updateTime, 1000);
});