document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
});

function addMessageToChat(sender, text) {
    const chatWindow = document.getElementById("chat-window");

    const bubble = document.createElement("div");
    bubble.style.margin = "10px 0";
    bubble.style.maxWidth = "75%";
    bubble.style.padding = "10px 14px";
    bubble.style.borderRadius = "12px";
    bubble.style.whiteSpace = "pre-wrap";

    if (sender === "user") {
        bubble.style.background = "#DCF8C6";
        bubble.style.marginLeft = "auto";
        bubble.style.textAlign = "right";
    } else {
        bubble.style.background = "#F1F0F0";
        bubble.style.marginRight = "auto";
        bubble.style.textAlign = "left";
    }

    bubble.innerText = text;
    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();
    if (!message) return;

    addMessageToChat("user", message);
    input.value = "";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });
        const data = await response.json();
        addMessageToChat("bot", data.answer || "(Error: No response from server)");
    } catch (err) {
        addMessageToChat("bot", `(Error: ${err})`);
    }
}