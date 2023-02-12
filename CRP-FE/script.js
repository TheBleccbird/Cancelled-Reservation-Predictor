loadIntoTable("functional_test_data.json", document.getElementById("table"));

async function loadIntoTable(url, table){
    const tableHead = table.querySelector("thead");
    const tableBody = table.querySelector("tbody");

    const response = await fetch(url)
    const {headers, rows} = await response.json();

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

    const response = await fetch(url, {
        method: 'POST',
        mode: 'cors',
        cache: 'no-cache',
        credentials: 'same-origin',
        headers: {
            'Content-Type': 'application/json'
        },
        redirect: 'follow',
        referrerPolicy: 'no-referrer',
        body: JSON.stringify(data)
    });
    return response.json();
}
