async function generateStats() {
    const name = document.getElementsByName('name')[0].value;
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const clothes = document.getElementById('clothes').value;
    const entry_time = document.getElementById('entry_time').value;
    const exit_time = document.getElementById('exit_time').value;

    if (!name || !age || !gender || !clothes || !entry_time || !exit_time) {
        alert("Barcha maydonlarni to'ldiring!");
        return;
    }

    const res = await fetch('/generate_statistics', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            name, age, gender, clothes, entry_time, exit_time
        })
    });

    if (!res.ok) {
        const err = await res.json();
        alert(err.error || "Statistika generatsiyasida xatolik!");
        return;
    }

    const data = await res.json();

    // Update UI with both predictions
    document.getElementById('heuristicProb').innerHTML =
        `<b>Heuristic model taxmini:</b> ${(data.purchase_prob).toFixed(1)}%`;
    document.getElementById('mlProb').innerHTML =
        `<b>ML model taxmini:</b> ${(data.ml_prediction * 100).toFixed(1)}%`;
    document.getElementById('finalProb').innerHTML =
        `<b>Umumiy xarid ehtimoli:</b> ${(data.final_probability * 100).toFixed(1)}%`;

    // ... (rest of your existing UI updates) ...
}