#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QVBoxLayout>

using namespace mlpack;
using namespace mlpack::data;
using namespace arma;

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	ui->resultLabel->setScaledContents(true);
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::setResultText(const QString &text)
{
	ui->resultLabel->setText(text);
	ui->resultLabel->adjustSize();
}

void MainWindow::setDataset(const arma::mat &data)
{
	qDebug() << data.n_rows;
	ui->datasetTable->setRowCount(data.n_rows);
	ui->datasetTable->setColumnCount(data.n_rows);
	for(int col = 0; col < data.n_cols; ++col)
	{
		for(int row = 0; row < data.n_rows; ++row)
		{
			QTableWidgetItem *elem = new QTableWidgetItem;
			elem->setText(QString("%1").arg(data.at(row, col)));
			ui->datasetTable->setItem(row, col, elem);
			QTableWidgetItem *colHeadElem = new QTableWidgetItem;
			colHeadElem->setText(QString("%1").arg(col));
			ui->datasetTable->setHorizontalHeaderItem(col, colHeadElem);

			QTableWidgetItem *rowHeadElem = new QTableWidgetItem;
			rowHeadElem->setText(QString("%1").arg(row));
			ui->datasetTable->setHorizontalHeaderItem(row, rowHeadElem);
		}
	}
	ui->datasetTable->adjustSize();
}

