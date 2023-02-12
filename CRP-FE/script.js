loadIntoTable("functional_test_data.json", document.getElementById("table"));

async function loadIntoTable(url, table){
    const tableHead = table.querySelector("thead");
    const tableBody = table.querySelector("tbody");

    const response = await fetch(url)
    const {headers, rows} = await response.json();
    
    //clear table

    tableHead.innerHTML = "<tr></tr>";
    tableBody.innerHTML = "";

    for (const headerText of headers)
    {
        const headerElement = document.createElement("th");
        
        headerElement.textContent = headerText;
        tableHead.querySelector("tr").appendChild(headerElement);
        
    }

    for (const row of rows)
    {
        const rowElement = document.createElement("tr");
        
        const prediction = await postData('http://127.0.0.1:5000/predict', row);
        const predictionCell = document.createElement("td");
        predictionCell.className = "predictionCell";
        
        if(prediction.response)
        {
            predictionCell.textContent = "Yes";    
            predictionCell.style.color = "red";
        }
        else
        {
            predictionCell.textContent = "No";
            predictionCell.style.color = "green";    
        }

        

        rowElement.appendChild(predictionCell);

        for (const cellText of Object.values(row)){
            const cellElement = document.createElement("td");

            cellElement.textContent = cellText;
            rowElement.appendChild(cellElement);
        }

        tableBody.appendChild(rowElement);
    }

}

async function postData(url = '', data = {}) {
    // Default options are marked with *
    const response = await fetch(url, {
        method: 'POST', // *GET, POST, PUT, DELETE, etc.
        mode: 'cors', // no-cors, *cors, same-origin
        cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
        credentials: 'same-origin', // include, *same-origin, omit
        headers: {
            'Content-Type': 'application/json'
            // 'Content-Type': 'application/x-www-form-urlencoded',
        },
        redirect: 'follow', // manual, *follow, error
        referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
        body: JSON.stringify(data) // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects
}
