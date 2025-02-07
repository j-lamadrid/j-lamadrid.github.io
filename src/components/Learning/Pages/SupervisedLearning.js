import React from 'react';
import { Container, Accordion } from 'react-bootstrap';
import Particle from '../../Particle';
import '../../../style.css';
import { JupyterNotebookViewer } from "react-jupyter-notebook-viewer";

const SupervisedLearning = () => {
    const renderNotebook = (notebookSrc) => {
        return (
            <JupyterNotebookViewer
			filePath={notebookSrc}
			notebookInputLanguage="python"
		/>
        );
    };

    return (
        <Container fluid className="learning-section">
            <Particle />
            <Container>
                <h1 className="heading">Supervised Learning</h1>
                <p className="description">
                    Welcome to the Supervised Learning page. Here you can find interactive Jupyter notebooks to help you learn more about supervised learning techniques.
                </p>
                <Accordion>
                    <Accordion.Item eventKey="0" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">Regression</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            {renderNotebook(process.env.PUBLIC_URL + "/notebooks/Stonks.ipynb")}
                        </Accordion.Body>
                    </Accordion.Item>
                    <Accordion.Item eventKey="1" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">Support Vector Machine (SVM)</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            {renderNotebook("/path/to/svm_notebook.html")}
                        </Accordion.Body>
                    </Accordion.Item>
                    <Accordion.Item eventKey="2" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">K-Nearest Neighbors (KNN)</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            {renderNotebook("/path/to/knn_notebook.html")}
                        </Accordion.Body>
                    </Accordion.Item>
                </Accordion>
            </Container>
        </Container>
    );
};

export default SupervisedLearning;
