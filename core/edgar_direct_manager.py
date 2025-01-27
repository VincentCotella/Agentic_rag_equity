# core/edgar_direct_manager.py

import logging
from edgar import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgarDirectManager:
    """
    Gère la récupération du dernier 10-K pour un ticker, 
    et expose des méthodes pour récupérer le contenu d'Items spécifiques.
    """

    def __init__(self, 
                 ticker_symbol: str, 
                 user_agent: str = "Vincent Cotella vincent.cotella@edu.devinci.fr", 
                 report_type="10-K"):
        self.ticker_symbol = ticker_symbol.upper()
        self.report_type = report_type
        set_identity(user_agent)

    def _get_full_filing_obj(self):
        """
        Renvoie l'objet subscriptable (par ex. filing_obj["Item 1"]).
        """
        company = Company(self.ticker_symbol)
        filing = company.get_filings(form=self.report_type).latest(1)
        logger.info(f"Récupération du dernier {self.report_type} pour {self.ticker_symbol}")
        return filing.obj()  # p. ex. TenK object subscriptable

    def get_item_text(self, item_label: str) -> str:
        """
        Renvoie le texte complet d'un Item (ex. "Item 1"), 
        ou "Not Available" si la clé n'existe pas.
        """
        filing_obj = self._get_full_filing_obj()
        try:
            return filing_obj[item_label]
        except KeyError:
            return "Not Available"

    def get_items_concat(self, item_labels: list) -> str:
        """
        Concatène le texte de plusieurs Items en un seul string,
        séparés par 2 sauts de ligne.
        """
        filing_obj = self._get_full_filing_obj()
        chunks = []
        for label in item_labels:
            try:
                text = filing_obj[label]
            except KeyError:
                text = "Not Available"
            chunks.append(text)
        return "\n\n".join(chunks)
